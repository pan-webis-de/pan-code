import argparse
import json
from typing import List, Dict

from sklearn.metrics import f1_score, matthews_corrcoef

from span_f1_metric import compute_score_pr

SPAN_EMPTY_LABEL = 'X'

SPAN_LABELS = [
    'OBJECTIVE', 'AGENT', 'FACILITATOR', 'CAMPAIGNER', 'VICTIM', 'NEGATIVE_EFFECT'
]

def load_json(file_path):
    '''
    Load a JSON file from the given path.
    :param file_path: Path to the JSON file.
    :return: The loaded JSON object.
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def ensure_id_uniqueness(json_data):
    '''
    Ensure that the IDs in the predictions are unique.
    :return:
    '''
    ids = set()
    for txt_data in json_data:
        if txt_data['id'] in ids:
            raise ValueError(f"ID {txt_data['id']} is not unique!")
        ids.add(txt_data['id'])

def ensure_id_equality(predictions, gold):
    '''
    Ensure that the predictions and gold labels have the same IDs.
    :return:
    '''
    pred_ids = set(txt_data['id'] for txt_data in predictions)
    gold_ids = set(txt_data['id'] for txt_data in gold)
    if pred_ids != gold_ids:
        raise ValueError(f"IDs in predictions and gold labels do not match!")

def ensure_id_consistency(predictions, gold):
    '''
    Ensure that the predictions and gold labels have the same IDs.
    :return:
    '''
    ensure_id_uniqueness(predictions)
    ensure_id_uniqueness(gold)
    ensure_id_equality(predictions, gold)

def check_binary_predicitons(predictions):
    '''
    Ensure that the prediction texts have a 'category' field, and that the category is valid
    '''
    for txt_data in predictions:
        if 'category' not in txt_data:
            raise ValueError(f"Category field missing in prediction with ID {txt_data['id']}")
        if txt_data['category'] not in ['CONSPIRACY', 'CRITICAL']:
            raise ValueError(f"Invalid category '{txt_data['category']}' in prediction with ID {txt_data['id']}")

def check_sequence_annotations(annotations: List[Dict]) -> bool:
    '''
    Check that the annotations are in a valid format.
    return: True if the annotations are empty, False otherwise
    '''
    if not isinstance(annotations, list):
        raise ValueError(f"Annotations must be a list, got {type(annotations)}")
    if len(annotations) == 0: return True # no annotations
    if not all(isinstance(ann, dict) for ann in annotations):
        raise ValueError(f"Annotations must be a list of dictionaries")
    # if there exists an empty annotation, with category SPAN_EMPTY_LABEL, it must be the only one
    empty_anns = [ann for ann in annotations if ann['category'] == SPAN_EMPTY_LABEL]
    non_empty_anns = [ann for ann in annotations if ann['category'] != SPAN_EMPTY_LABEL]
    if len(empty_anns) > 1 or (len(empty_anns) == 1 and len(non_empty_anns) > 0):
        raise ValueError(f"Only one empty annotation allowed, and it must be the only annotation, "
                         f"instead the annotations are: {annotations}")
    if len(empty_anns) == 1: return True # no annotations except the empty one
    # check fields of non-empty annotations
    for ann in annotations:
        # must contain 'start', 'end', 'category' fields
        if not all(key in ann for key in ['start_char', 'end_char', 'category']):
            raise ValueError(f"Annotation missing required fields (start, end, category): {ann}")
        # 'start' and 'end' must be integers
        if not all(isinstance(ann[key], int) for key in ['start_char', 'end_char']):
            raise ValueError(f"Annotation 'start_char' and 'end_char' must be integers: {ann}")
        # 'category' must be a valid label
        if ann['category'] not in SPAN_LABELS:
            raise ValueError(f"Invalid category '{ann['category']}' in annotation: {ann}")
    return False

def check_sequence_predictions(predictions):
    '''
    Ensure that the prediction texts have a 'annotations' field, and that the annotations are valid
    :param predictions:
    :return:
    '''
    for txt_data in predictions:
        if 'annotations' not in txt_data:
            raise ValueError(f"Annotations field missing in prediction with ID {txt_data['id']}")
        annotations = txt_data['annotations']
        check_sequence_annotations(annotations)


def extract_binary_labels(predictions: List[Dict], gold: List[Dict],
                          positive_class = 'CONSPIRACY') -> (List[int], List[int]):
    '''
    Extract 0/1 labels from the predictions and gold labels, aligned by ID.
    Assumes ID sets are equal, and that the IDs are unique.
    '''
    pred_labels = []
    gold_labels = []
    gold_dict = {txt_data['id']: txt_data for txt_data in gold}
    for txt_data in predictions:
        pred_labels.append(1 if txt_data['category'] == positive_class else 0)
        gold_labels.append(1 if gold_dict[txt_data['id']]['category'] == positive_class else 0)
    return pred_labels, gold_labels

def evaluate_task1(predictions_path, gold_path, verbose=False):
    '''
    Evaluate the predictions for Task 1, the binary classification task.
    :return:
    '''
    # load and validate the predictions
    predictions = load_json(predictions_path)
    gold = load_json(gold_path)
    ensure_id_consistency(predictions, gold)
    check_binary_predicitons(predictions)
    # calculate and print the evaluation metrics
    pred_labels, gold_labels = extract_binary_labels(predictions, gold)
    mcc = matthews_corrcoef(gold_labels, pred_labels)
    f1_pos = f1_score(gold_labels, pred_labels)
    f1_neg = f1_score([1 - label for label in gold_labels], [1 - label for label in pred_labels])
    f1_macro = f1_score(gold_labels, pred_labels, average='macro')
    if verbose:
        print(f"MCC: {mcc:.3f}, F1 (macro): {f1_macro:.3f}, F1 (conspi): {f1_pos:.3f}, F1 (critical): {f1_neg:.3f}")
    # return a map of the evaluation metrics, with key-value pairs
    return {'MCC': mcc, 'F1-macro': f1_macro, 'F1-conspiracy': f1_pos, 'F1-critical': f1_neg}

def spans_annots_to_spanF1_format(texts_json: List[Dict]) -> Dict[str, List[List]]:
    '''
    Convert span annotations loaded from .json to the format used by the spanF1 scorer:
    a map text_id -> list of spans in span-f1 format,
    where each span is a list[label, set[char_indices of the span]]
    '''
    result = {}
    for annot_text in texts_json:
        text_id = annot_text['id']
        # clean empty spans
        spans_json = [span for span in annot_text['annotations'] if span['category'] != SPAN_EMPTY_LABEL]
        spans = [(s['category'], s['start_char'], s['end_char']) for s in spans_json]
        if text_id not in result: result[text_id] = []
        labels = sorted(list(set([s[0] for s in spans])))
        f1spans = []
        for l in labels:
            # take all spans with label l, and sort them by the start index
            span_ranges = sorted([s[1:3] for s in spans if s[0] == l], key=lambda x: x[0])
            # map each char range to a set of character indices
            for start, end in span_ranges:
                f1spans.append([l, set(range(start, end+1))])
        result[text_id] = f1spans
    return result

def calc_macro_averages(result, verbose=False, overwrite_with_macro=False):
    '''
    Complete the results of the spanF1 scorer (compute_score_pr) with macro-averages.
    Calculate macro-averages for precision, recall, and F1.
    Assign 'micro' prefix to the current values.
    '''
    measures = ['P', 'R', 'F1']
    for measure in measures:
        result[f'micro-{measure}'] = result[measure]
        values = [result[f'{label}-{measure}'] for label in SPAN_LABELS]
        result[f'macro-{measure}'] = sum(values) / len(values)
    if overwrite_with_macro: # substitute the (micro) P, R, and F1 values with the macro values
        for measure in measures: result[measure] = result[f'macro-{measure}']
    if verbose: # print all the results - micro, macro, and per-label
        for measure in measures:
            print(f"{measure}: {result[f'micro-{measure}']:.3f} (micro), {result[f'macro-{measure}']:.3f} (macro)")
            for label in SPAN_LABELS:
                print(f"{label}-{measure}: {result[f'{label}-{measure}']:.3f}")
            print()

def evaluate_task2(predictions_path, gold_path, verbose=False):
    '''
    Evaluate the predictions for Task 2, the sequence labeling task.
    :return:
    '''
    # load and validate the predictions
    predictions = load_json(predictions_path)
    gold = load_json(gold_path)
    ensure_id_consistency(predictions, gold)
    check_sequence_predictions(predictions)
    # calculate and print the evaluation metrics
    predictions_spanf1 = spans_annots_to_spanF1_format(predictions)
    gold_spanf1 = spans_annots_to_spanF1_format(gold)
    result = compute_score_pr(predictions_spanf1, gold_spanf1, SPAN_LABELS, disable_logger=True)
    calc_macro_averages(result, verbose=verbose)
    if verbose:
        print(f"Span F1: {result['macro-F1']:.3f}, Span P: {result['macro-P']:.3f}, Span R: {result['macro-R']:.3f}")
    # return a map of key-value pairs of the evaluation metrics, only for the above metrics
    return {f'span-{measure}': result[f'macro-{measure}'] for measure in ['P', 'R', 'F1']}

def print_results(results, outdir):
    '''
    Output results in PAN/tira format.
    results: a dictionary of evaluation metrics (name -> value)
    outdir: the directory where the results will be saved in the file 'evaluation.prototext'
    '''
    if not outdir.endswith('/'): outdir += '/'
    with open(outdir + 'evaluation.prototext', 'w', encoding='utf-8') as f:
        for metric, score in results.items():
            f.write('measure {\n')
            f.write(' key: "' + metric + '"\n')
            f.write(' value: "' + str(score) + '"\n')
            f.write('}\n')

def main():
    '''
    Entry point for command line evaluation interface.
    '''
    parser = argparse.ArgumentParser(description="Evaluator of the PAN 2024 'Oppositional' task.")
    # Required arguments
    parser.add_argument("task", type=str, help="Task to evaluate", choices=["task1", "task2"])
    parser.add_argument("--predictions", required=True, type=str, help="Path of the .json file with the predictions")
    parser.add_argument("--gold", required=True, type=str, help="Path of the .json file with the gold (ground truth) labels")
    parser.add_argument("--outdir", required=True, type=str, help="Path to the folder where the results will be saved")
    args = parser.parse_args()
    if args.task == "task1":
        res = evaluate_task1(args.predictions, args.gold)
    elif args.task == "task2":
        res = evaluate_task2(args.predictions, args.gold)
    print_results(res, args.outdir)

if __name__ == '__main__':
    main()