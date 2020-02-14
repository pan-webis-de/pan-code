#!/usr/bin/python3
import argparse
import json
from typing import Tuple
from collections import Counter
from statistics import mean


def parse_input() -> Tuple[list, list]:
    """
    read the files given as parameters and load the newline-delimited json stings
    :return: a tuple with   [0] a list of dicts with predicted classes,
                            [1] a list of dicts with the true classes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", help="path to the predicted labels.ndjson")
    parser.add_argument("-t", "--truth", help="path to the true labels.ndjson")
    args = parser.parse_args()
    return ([json.loads(u) for u in open(args.predictions).readlines()],
            [json.loads(u) for u in open(args.truth).readlines()])


def harmonic_mean(l: list):
    """
    calculate the harmonic mean of a list of classes
    :param l: a list holding elements
    :return:
    """
    return len(l) / sum([1 / x for x in l])


def mc_prec_rec(mc_p: list, mc_t: list, hit_function=lambda x, y: x == y) -> Tuple[list, list]:
    """
    computes multi value recall and precision. Indices of inputs must match.
    :param hit_function: function to calculate true positives
    :param mc_p: list of predicted values
    :param mc_t: list of true values.
    :return: tuple: list of precision for classes, list of recall for class
    """
    def safe_divide(x, y):
        return x / y if y != 0 else 0

    true_positive = [t for p, t in zip(mc_p, mc_t) if hit_function(p, t)]
    false_positives = [p for p, t in zip(mc_p, mc_t) if not hit_function(p, t)]
    positive_in_prediction = true_positive + false_positives
    positive_in_truth = Counter(mc_t)

    tp_c = Counter(true_positive)
    pp_c = Counter(positive_in_prediction)

    precisions = [safe_divide(tp_c.get(cls, 0), pp_c.get(cls, 0))
                  for cls in positive_in_truth.keys()]
    recalls = [tp_c.get(cls, 0) / positive_in_truth.get(cls, 0) for cls in positive_in_truth.keys()]

    return precisions, recalls


def age_window_hit(by_predicted, by_truth):
    """
    calculates the window for a given truth and checks if the prediction lies within that window
    :param by_predicted: the predicted birth year
    :param by_truth: the true birth year
    :return: true if by_predicted within m-window of by_truth
    """
    m = -0.1 * by_truth + 202.8
    return int(by_truth - m) <= by_predicted <= int(by_truth + m)


if __name__ == "__main__":
    """
    This is the evaluator for the PAN@CLEF19 Task "Celebrity Profiling"
    It outputs 5 Metrics: 
      - cRank, the harmonic mean of the sub metrics below. This is the primary metric. 
      - F1_gender, the harmonic mean of the average multi class precision and recall for gender prediction
      - F1_occupation, same as above for occupation
      - F1_fame, same as above for fame
      - F1_age, same as above, but positives are lenient in a window around the prediction
      
    For more information visit: 
      - https://pan.webis.de/clef19/pan19-web/celebrity-profiling.html
      
    Please send any requests or remarks to:
      - matti.wiegmann@uni-weimar.de
      - pan@webis.de
    """
    predictions, truth = parse_input()

    gender_prec, gender_rec = mc_prec_rec([u["gender"] for u in predictions],
                                          [u["gender"] for u in truth])
    occ_prec, occ_rec = mc_prec_rec([u["occupation"] for u in predictions],
                                    [u["occupation"] for u in truth])
    fame_prec, fame_rec = mc_prec_rec([u["fame"] for u in predictions],
                                      [u["fame"] for u in truth])
    age_prec, age_rec = mc_prec_rec([u["birthyear"] for u in predictions],
                                    [u["birthyear"] for u in truth], hit_function=age_window_hit)

    F1_gender = harmonic_mean([mean(gender_prec), mean(gender_rec)])
    F1_occupation = harmonic_mean([mean(occ_prec), mean(occ_rec)])
    F1_fame = harmonic_mean([mean(fame_prec), mean(fame_rec)])
    F1_age = harmonic_mean([mean(age_prec), mean(age_rec)])

    print("cRank: ", harmonic_mean([F1_gender, F1_occupation, F1_fame, F1_age]))
    print("F1_gender: ", F1_gender)
    print("F1_occupation: ", F1_occupation)
    print("F1_fame: ", F1_fame)
    print("F1_age: ", F1_age)
