from collections import defaultdict
import sys
from pan15_text_alignment_evaluator_character_level import extract_annotations_from_files, case_recall, true_detections


def get_annotations_per_document_pair(annotations):
    annotations_per_document_pair = defaultdict(set)
    for annotation in annotations:
        pair = (annotation.this_reference, annotation.source_reference)
        annotations_per_document_pair[pair].add(annotation)
    return annotations_per_document_pair


def precision(detection, cases):
    return recall(detection, cases)


def recall(case, detections):
    return case_recall(case, detections)


def recall_condition(s, R, tau_recall):
    rec = recall(s, R)
    return rec >= tau_recall


def precision_condition(s, S, R, tau_precision):
    true_detects = set(true_detections([s, ], R))
    for r in true_detects:
        prec = precision(r, S)
        if prec >= tau_precision:
            return True
    return False


def recall_condition1(r, S, tau_recall):
    true_detects = set(true_detections(S, [r, ]))
    for s in S:
        rec = recall(s, true_detects)
        if rec >= tau_recall:
            return True
    return False

def precision_condition1(r, S, tau_precision):
    prec = precision(r, S)
    return prec >= tau_precision


def case_level_performance(plag_path, det_path, pairs_file, tau_recall, tau_precision):
    S = extract_annotations_from_files(plag_path, "plagiarism")  # cases
    R = extract_annotations_from_files(det_path, "detected-plagiarism")  # detections

    S_dash = [s for s in S if recall_condition(s, R, tau_recall) and precision_condition(s, S, R, tau_precision)]
    R_dash = [r for r in R if precision_condition1(r, S, tau_precision) and recall_condition1(r, S, tau_recall)]

    rec = float(len(S_dash)) / len(S)
    prec = 0 if len(R) == 0 else float(len(R_dash)) / len(R)
    fmeas = get_fmeasure(prec, rec)

    return (prec, rec, fmeas)


def get_fmeasure(precision, recall, beta=1.0):
    nominator = (1 + beta * beta) * precision * recall
    denominator = (beta * beta * precision) + recall
    return 0 if denominator == 0 else float(nominator) / denominator


def main():
    plag_path = sys.argv[1]
    pairs_file = plag_path+"/pairs"
    det_path = sys.argv[2]
    tau_precision = float(sys.argv[3])
    tau_recall = float(sys.argv[4])

#     #===========================================================================
#     plag_path = "/mnt/nfs/tira/data/pan14-test-corpora-truth/pan14-text-alignment-test-corpus2-2014-05-09/"
#     pairs_file = plag_path + "pairs"
#     det_path = "/mnt/nfs/tira/runs/pan14-text-alignment-test-corpus2-2014-05-09/gillam14/2014-05-15-17-31-12/output"
#     tau_recall = 0.5
#     tau_precision = 0.5
#     #===========================================================================

    # extract information from detection path
    user = det_path.split("/")[-3]
    input_run = det_path.split("/")[-2]

    prec, rec, fmeasure = case_level_performance(plag_path, det_path, pairs_file, tau_recall, tau_precision)
    print "%s\t%s\t%0.5f\t%0.5f\t%0.5f" % (user, input_run, prec, rec, fmeasure)


if __name__ == '__main__':
    main()
    sys.exit(0)
