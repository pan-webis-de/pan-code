import argparse
import json
import os
import string
import subprocess

from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
from CB_Tokenizers import LemmaTokenizer

def mode_huggingface(args):
    with open(args.preds, 'r', encoding='utf-8') as al:
        pred_dict = json.load(al)

    truths = [json.loads(line) for line in open(args.truth, 'r', encoding='utf-8')]

    print('Writing uuids, truths.txt and predictions.txt ...')
    with open('uuids.txt', 'w', encoding='utf-8') as u:
        with open('truths.txt', 'w', encoding='utf-8') as t:
            with open('preds.txt', 'w', encoding='utf-8') as p:

                pred_uuids = pred_dict.keys()
                for truth in truths:
                    if truth['uuid'] in pred_uuids:

                        u.write(truth['uuid'] + '\n')
                        t.write(truth['spoiler'][0] + '\n')
                        p.write(pred_dict[truth['uuid']][0]['text'] + '\n')

def mode_albert(args):
    with open(args.preds, 'r', encoding='utf-8') as al:
        pred_dict = json.load(al)

    truths = [json.loads(line) for line in open(args.truth, 'r', encoding='utf-8')]

    print('Writing uuids, truths.txt and predictions.txt ...')
    with open('uuids.txt', 'w', encoding='utf-8') as u:
        with open('truths.txt', 'w', encoding='utf-8') as t:
            with open('preds.txt', 'w', encoding='utf-8') as p:

                pred_uuids = pred_dict.keys()
                for truth in truths:
                    if truth['uuid'] in pred_uuids:

                        u.write(truth['uuid'] + '\n')
                        t.write(truth['spoiler'][0] + '\n')
                        p.write(pred_dict[truth['uuid']]['text'] + '\n')

def mode_allenai(args):
    preds = {}
    with open(args.preds, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)

            preds[obj['uuid']] = obj

    truths = [json.loads(line) for line in open(args.truth, 'r', encoding='utf-8')]

    print('Writing uuids, truths.txt and predictions.txt ...')
    with open('uuids.txt', 'w', encoding='utf-8') as u:
        with open('truths.txt', 'w', encoding='utf-8') as t:
            with open('preds.txt', 'w', encoding='utf-8') as p:

                pred_uuids = preds.keys()
                for truth in truths:
                    if truth['uuid'] in pred_uuids:

                        u.write(truth['uuid'] + '\n')
                        t.write(truth['spoiler'][0] + '\n')
                        p.write(preds[truth['uuid']]['allenai_answer'] + '\n')

def bleu_score():
    tokenizer = LemmaTokenizer(lower=True)

    def stopfilter(tokens):
        tmp = [token for token in tokens if token not in stopwords.words('english')]
        res = [token for token in tmp if token not in string.punctuation]
        return res

    def make_score(trut, predi):
        if len(trut) > 3 and len(predi) > 3:
            weights = (1./4., 1./4., 1./4., 1./4.)
        elif len(trut) > 2 and len(predi) > 2:
            weights = (1./3., 1./3., 1./3.)
        elif len(trut) > 1 and len(predi) > 1:
            weights = (1./2., 1./2.)
        else:
            weights = (1., 0.)

        if (len(weights) == 4) and (len(trut) < 4 or len(predi) < 4):
            print(trut)
            print(predi)
            print(weights)
            print('\n')

        return sentence_bleu([trut], predi, weights=weights)

    truth = open('truths.txt', 'r', encoding='utf-8').readlines()
    prediction = open('preds.txt', 'r', encoding='utf-8').readlines()
    uuids = open('uuids.txt', 'r', encoding='utf-8').readlines()

    score = 0.
    lem_score = 0.

    write_dict = {'single_scores': {}, 'scores': {}}

    for i in range(len(truth)):
        if i % 100 == 0:
            print(f'\t{i}/{len(truth)}', end='\r')

        real_answer = truth[i].replace('\n', '')
        pred_answer = prediction[i].replace('\n', '')
        uuid = uuids[i].replace('\n', '')
        lem_truth_tokens = stopfilter(tokenizer(real_answer))
        lem_prediction_tokens = stopfilter(tokenizer(pred_answer))
        i_lem_score = make_score(lem_truth_tokens, lem_prediction_tokens)
        lem_score += i_lem_score
        write_dict['single_scores'][uuid] = {"truth": real_answer, "prediction": pred_answer, "bleu4_score": i_lem_score}

    lem_score /= len(truth)

    write_dict['scores']['bleu4_lemma'] = lem_score

    return write_dict

def bleu_score_param(truthtxt, predtxt):
    tokenizer = LemmaTokenizer(lower=True)

    def stopfilter(tokens):
        tmp = [token for token in tokens if token not in stopwords.words('english')]
        res = [token for token in tmp if token not in string.punctuation]
        return res

    def make_score(trut, predi):
        if len(trut) > 3 and len(predi) > 3:
            weights = (1./4., 1./4., 1./4., 1./4.)
        elif len(trut) > 2 and len(predi) > 2:
            weights = (1./3., 1./3., 1./3.)
        elif len(trut) > 1 and len(predi) > 1:
            weights = (1./2., 1./2.)
        else:
            weights = (1., 0.)

        if (len(weights) == 4) and (len(trut) < 4 or len(predi) < 4):
            print(trut)
            print(predi)
            print(weights)
            print('\n')

        return sentence_bleu([trut], predi, weights=weights)

    truth = open(truthtxt, 'r', encoding='utf-8').readlines()
    prediction = open(predtxt, 'r', encoding='utf-8').readlines()

    score = 0.
    lem_score = 0.

    write_dict = {'single_scores': {}}

    for i in range(len(truth)):
        if i % 100 == 0:
            print(f'\t{i}/{len(truth)}', end='\r')

        real_answer = truth[i].replace('\n', '')
        pred_answer = prediction[i].replace('\n', '')
        # truth_tokens = [token for token in word_tokenize(real_answer) if token not in string.punctuation]
        # prediction_tokens = [token for token in word_tokenize(pred_answer) if token not in string.punctuation]
        # i_score = make_score(truth_tokens, prediction_tokens)
        # score += i_score

        lem_truth_tokens = stopfilter(tokenizer(real_answer))
        lem_prediction_tokens = stopfilter(tokenizer(pred_answer))
        i_lem_score = make_score(lem_truth_tokens, lem_prediction_tokens)
        lem_score += i_lem_score
        write_dict['single_scores'][i] = {"truth": real_answer, "prediction": pred_answer,
                                             "bleu4_score": i_lem_score}

    # score /= len(truth)
    lem_score /= len(truth)

    # print("BLEU on plain: " + str(score))
    print("BLEU on lemmatized, stopped: " + str(lem_score))

    with open('bleu4_scores.json', 'w', encoding='utf-8') as write_file:
        json.dump(write_dict, write_file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth', help="Path to test .jsonl")
    parser.add_argument('--preds', help="Path to predictions file")
    parser.add_argument('--uuids', help="uuids file to write")
    parser.add_argument('--output_dir', help="Directory to write output files to")
    parser.add_argument('--meteor_dir', help="Directory of meteor-1.5.jar")
    parser.add_argument('--mode', choices=['allenai', 'albert', 'huggingface', 'blank'],
                        help='choose mode ( albert | allenai | huggingface | blank)')
    args = parser.parse_args()

    os.chdir(str(args.output_dir))

    print(args.mode)
    if args.mode == 'albert':
        mode_albert(args)
    elif args.mode == 'allenai':
        mode_allenai(args)
    elif args.mode == 'huggingface':
        mode_huggingface(args)
    elif args.mode == 'blank':
        print('Calculating BLEU4 ...')
        bleu_score_param(args.truth, args.preds)
    else:
        raise Exception('Only modes available: albert | allenai | huggingface | blank')

    cmd = ['java', '-Xmx2G', '-jar', str(args.meteor_dir) + 'meteor-1.5.jar', 'truths.txt', 'preds.txt', '-l', 'en', '-norm', '-t', 'adq']

    print('Calculating METEOR ...')
    with open('meteor_scores.txt', 'w') as out:
        return_code = subprocess.call(cmd, stdout=out)

    if not args.mode == 'blank':
        print('Calculating BLEU4 ...')
        bleu_score()

    print('DONE!')
