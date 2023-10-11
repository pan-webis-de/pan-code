#!/usr/bin/env python3

__credits__ = ['David Dale', 'Daniil Moskovskiy', 'Dmitry Ustalov']

import argparse
import sys

import numpy as np
import torch
from sacrebleu import CHRF
from tqdm.auto import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def prepare_target_label(model, target_label):
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.')
    return target_label


def classify_texts(
        model,
        tokenizer,
        texts,
        second_texts=None,
        target_label=None,
        batch_size=32,
        raw_logits=False,
        desc=None
):
    target_label = prepare_target_label(model, target_label)
    res = []
    for i in trange(0, len(texts), batch_size, desc=desc):
        inputs = [texts[i: i + batch_size]]
        if second_texts is not None:
            inputs.append(second_texts[i: i + batch_size])
        inputs = tokenizer(
            *inputs, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(model.device)
        with torch.no_grad():
            try:
                logits = model(**inputs).logits
                if raw_logits:
                    preds = logits[:, target_label]
                elif logits.shape[-1] > 1:
                    preds = torch.softmax(logits, -1)[:, target_label]
                else:
                    preds = torch.sigmoid(logits)[:, 0]
                preds = preds.cpu().numpy()
            except:
                print(i, i + batch_size)
                preds = [0] * len(inputs)
        res.append(preds)
    return np.concatenate(res)


def evaluate_style(
        model,
        tokenizer,
        texts,
        target_label=1,  # 1 is formal, 0 is informal
        batch_size=32
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        texts,
        batch_size=batch_size,
        target_label=target_label,
        desc='Style'
    )
    return scores


def evaluate_meaning(
        model,
        tokenizer,
        original_texts,
        rewritten_texts,
        target_label="entailment",
        bidirectional=True,
        batch_size=32,
        aggregation="prod",
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        original_texts,
        rewritten_texts,
        batch_size=batch_size,
        target_label=target_label,
        desc='Meaning'
    )
    if bidirectional:
        reverse_scores = classify_texts(
            model,
            tokenizer,
            rewritten_texts,
            original_texts,
            batch_size=batch_size,
            target_label=target_label,
            desc='Meaning'
        )
        if aggregation == "prod":
            scores = reverse_scores * scores
        elif aggregation == "mean":
            scores = (reverse_scores + scores) / 2
        elif aggregation == "f1":
            scores = 2 * reverse_scores * scores / (reverse_scores + scores)
        else:
            raise ValueError('aggregation should be one of "mean", "prod", "f1"')
    return scores


def evaluate_cola(model, tokenizer, texts, target_label=1, batch_size=32):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        texts,
        batch_size=batch_size,
        target_label=target_label,
        desc='Fluency'
    )
    return scores


def evaluate_style_transfer(
        original_texts,
        rewritten_texts,
        style_model,
        style_tokenizer,
        meaning_model,
        meaning_tokenizer,
        fluency_model,
        fluency_tokenizer,
        references=None,
        style_target_label=1,
        meaning_target_label="paraphrase",
        cola_target_label=1,
        batch_size=32,
        aggregate=False,
        style_calibration=None,
        meaning_calibration=None,
        fluency_calibration=None,
):
    accuracy = evaluate_style(
        style_model,
        style_tokenizer,
        rewritten_texts,
        target_label=style_target_label,
        batch_size=batch_size,
    )
    similarity = evaluate_meaning(
        meaning_model,
        meaning_tokenizer,
        original_texts,
        rewritten_texts,
        batch_size=batch_size,
        bidirectional=False,
        target_label=meaning_target_label,
    )
    fluency = evaluate_cola(
        fluency_model,
        fluency_tokenizer,
        texts=rewritten_texts,
        batch_size=batch_size,
        target_label=cola_target_label,
    )

    if style_calibration:
        accuracy = style_calibration(accuracy)
    if meaning_calibration:
        similarity = meaning_calibration(similarity)
    if fluency_calibration:
        fluency = fluency_calibration(fluency)

    joint = accuracy * similarity * fluency

    result = dict(accuracy=accuracy, similarity=similarity, fluency=fluency, joint=joint)

    if references is not None:
        chrf_calc = CHRF()
        result["chrF"] = [
            chrf_calc.sentence_score(hyp, [ref]).score
            for hyp, ref in zip(rewritten_texts, references)
        ]

    if aggregate:
        return {k: float(np.mean(v)) for k, v in result.items()}

    return result


def load_model(
        model_name=None,
        model=None,
        tokenizer=None,
        model_class=AutoModelForSequenceClassification,
        use_cuda=True,
        use_auth_token=None,
):
    if model is None:
        if model_name is None:
            raise ValueError("Either model or model_name should be provided")
        model = model_class.from_pretrained(model_name, use_auth_token=use_auth_token)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either tokenizer or model_name should be provided")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=use_auth_token
        )
    return model, tokenizer


def format_prototext(measure: str, value: str) -> str:
    return f'measure{{\n  key: "{measure}"\n  value: "{value}"\n}}\n'


def run_evaluation(args, evaluator):
    original_texts = [line.strip() for line in args.input.readlines()][:10]
    rewritten_texts = [sentence.strip() for sentence in args.prediction.readlines()][:10]
    references = [line.strip() for line in args.golden.readlines()][:10]

    assert all(len(x) > 0 for x in original_texts)
    assert all(len(x) > 0 for x in rewritten_texts)
    assert all(isinstance(x, str) for x in original_texts)
    assert all(isinstance(x, str) for x in rewritten_texts)

    result = evaluator(original_texts, rewritten_texts, references)

    for measure in result.keys():
        args.output.write(format_prototext(measure, result[measure]))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=argparse.FileType('r', encoding='UTF-8'), required=True,
                        help='Initial texts before style transfer')
    parser.add_argument('-g', '--golden', type=argparse.FileType('r', encoding='UTF-8'), required=True,
                        help='Ground truth texts after style transfer')
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'), default=sys.stdout,
                        help='Where to print the evaluation results')
    parser.add_argument('--style-model', type=str, required=True,
                        help='Style evaluation model on Hugging Face Hub')
    parser.add_argument('--meaning-model', type=str, required=True,
                        help='Meaning evaluation model on Hugging Face Hub')
    parser.add_argument('--fluency-model', type=str, required=True,
                        help='Fluency evaluation model on Hugging Face Hub')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Whether to use CUDA acceleration, if possible')
    parser.add_argument('prediction', type=argparse.FileType('r', encoding='UTF-8'),
                        help='Your model predictions')

    args = parser.parse_args()

    style_model, style_tokenizer = load_model(args.style_model, use_cuda=args.cuda)
    meaning_model, meaning_tokenizer = load_model(args.meaning_model, use_cuda=args.cuda)
    fluency_model, fluency_tokenizer = load_model(args.fluency_model, use_cuda=args.cuda)

    def evaluator(original_texts, rewritten_texts, references):
        return evaluate_style_transfer(
            original_texts=original_texts,
            rewritten_texts=rewritten_texts,
            references=references,
            style_model=style_model,
            style_tokenizer=style_tokenizer,
            meaning_model=meaning_model,
            meaning_tokenizer=meaning_tokenizer,
            fluency_model=fluency_model,
            fluency_tokenizer=fluency_tokenizer,
            style_target_label=0,
            meaning_target_label=0,
            cola_target_label=0,
            aggregate=True,
        )

    run_evaluation(args, evaluator=evaluator)


if __name__ == '__main__':
    main()
