import json
import os

import click
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from tira.io_utils import to_prototext


def auc(true_y, pred_y):
    """
    Calculates the ROC-AUC score (Area Under the Curve).

    ROC-AUC is a well-known scalar evaluation score for binary classifiers.

    Parameters
    ----------
    true_y : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`.
    pred_y : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The Area Under the ROC Curve.
    """
    return roc_auc_score(true_y, pred_y)


def c_at_1(true_y, pred_y):
    """
    Calculates the c@1 score.

    This method rewards predictions which leave some problems
    unanswered (score = 0.5). See:
        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    Parameters
    ----------
    true_y : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`, unanswered problems = 0.5.
    pred_y : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The c@1 measure.
    """

    nu = pred_y == 0.5
    nc = np.sum((true_y == (pred_y > 0.5))[~nu])
    nu = np.sum(nu)
    return (1 / len(true_y)) * (nc + (nu * nc / len(true_y)))


def f1(true_y, pred_y):
    """
    Calculates the F1 score.

    Assesses verification performance, assuming that every score > 0.5
    represents a same-author pair decision.

    Parameters
    ----------
    true_y : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`.
    pred_y : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The F1 score.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """

    return f1_score(true_y, pred_y > 0.5)


def f05u(true_y, pred_y):
    """
    Calculates the F0.5u score.

    F0.5u is a modified F0.5 measure which treats unanswered problems as false negatives.
    as false negatives. See:
        J. Bevendorff et al. Generalizing unmasking for short texts.
        In Proc. of the 14th Conference of the North American Chapter of
        the Association for Computational Linguistics (NAACL 2019). Pages 654–659.

    Parameters
    ----------
    true_y : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`, unanswered problems = 0.5
    pred_y : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The F0.5u score.
    """

    n_tp = np.sum(true_y * (pred_y > 0.5))
    n_fn = np.sum(true_y * (pred_y < 0.5))
    n_fp = np.sum((1.0 - true_y) * (pred_y > 0.5))
    n_u = np.sum(pred_y == 0.5)

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)


def brier_score(true_y, pred_y):
    """
    Calculates the complement of the Brier score loss (which is bounded
    to [0-1]), so that higher scores indicate better performance.
    We use the Brier implementation in scikit-learn [Pedregosa et al. 2011].

    Parameters
    ----------
    true_y : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`.
    pred_y : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    Complement of the Brier score.
    """
    try:
        return 1 - brier_score_loss(true_y, pred_y)
    except ValueError:
        return 0.0


def load_problem_file(file_obj):
    problems = {}
    for l in file_obj:
        j = json.loads(l)

        if 'is_human' not in j:
            raise ValueError('Invalid input JSON schema')

        d = {k: v for k, v in j.items() if k not in ['id', 'is_human']}
        if type(j['is_human']) in [list, tuple]:
            assert j['is_human'][0] ^ j['is_human'][1]      # sanity check
            d['is_human'] = 0.0 if j['is_human'][0] else 1.0
        elif type(j['is_human']) in [int, float]:
            d['is_human'] = float(j['is_human'])
        else:
            raise ValueError(f'Invalid data type {type(j["is_human"])} for problem {j["id"]}."')
        problems[j['id']] = d

    return problems


def evaluate_all(true_y, pred_y):
    """
    Calculate all evaluation scores and return results as a dict.
    All scores are rounded to three digits.

    :param true_y : truth as numpy array [n_problems]
    :param pred_y : predictions as numpy array [n_problems]
    """

    results = {
        'roc-auc': auc(true_y, pred_y),
        'brier': brier_score(true_y, pred_y),
        'c@1': c_at_1(true_y, pred_y),
        'f1': f1(true_y, pred_y),
        'f05u': f05u(true_y, pred_y)
    }
    results['mean'] = np.mean(list(results.values()))

    for k, v in results.items():
        results[k] = round(v, 3)

    return results


def vectorize_and_evaluate(truth_dict, pred_dict, missing_default=0.5):
    """
    Vectorize input data into numpy arrays and evaluate them.

    :param truth_dict: ground truth dict
    :param pred_dict: predictions dict
    :param missing_default: default missing predictions to this value
    :return: dict with evaluation results
    """
    missing_default = {'is_human': missing_default}
    scores = [(truth_dict[k]['is_human'], pred_dict.get(k, missing_default)['is_human']) for k in sorted(truth_dict)]
    truth, pred = zip(*scores)
    truth = np.array(truth, dtype=np.float64)
    pred = np.clip(np.array(pred, dtype=np.float64), 0.0, 1.0)
    assert len(truth) == len(pred)
    return evaluate_all(truth, pred)


@click.command(help='Evaluation script for GenAI Authorship Verification @ PAN\'24')
@click.argument('answer_file', type=click.File('r'))
@click.argument('truth_file', type=click.File('r'))
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--outfile-name', default='evaluation.json', show_default=True,
              help='Output JSON filename')
@click.option('-p', '--skip-prototext', is_flag=True,
              help='Skip Tira Prototext output, only write JSON')
@click.option('-s', '--skip-source-eval', is_flag=True, help='Skip evaluation of individual sources')
def main(answer_file, truth_file, output_dir, outfile_name, skip_prototext, skip_source_eval):
    pred = load_problem_file(answer_file)
    truth = load_problem_file(truth_file)

    click.echo(f'-> {len(truth)} problems in ground truth', err=True)
    click.echo(f'-> {len(pred)} solutions explicitly proposed', err=True)

    if len(pred) > len(truth) or set(truth.keys()).union(set(pred)) != set(truth.keys()):
        raise click.UsageError('Truth file does not match answer file.')

    # Evaluate all test cases
    results = vectorize_and_evaluate(truth, pred)

    # Write Tira Prototext
    if not skip_prototext:
        with open(os.path.join(output_dir, os.path.splitext(outfile_name)[0] + '.prototext'), 'w') as f:
            f.write(to_prototext([results]))

    # Evaluate test cases for individual sources and add to JSON output
    sources = sorted({v['source_id'].split('/', 1)[0] for v in truth.values() if 'source_id' in v})
    if sources and not skip_source_eval:
        results['_sources'] = {}
        for s in sources:
            t = {k: v for k, v in truth.items() if v['source_id'].startswith(s + '/')}
            results['_sources'][s] = vectorize_and_evaluate(t, pred)

    jstr = json.dumps(results, indent=4, sort_keys=False)
    click.echo(jstr)
    with open(os.path.join(output_dir, outfile_name), 'w') as f:
        f.write(jstr)


if __name__ == '__main__':
    main()
