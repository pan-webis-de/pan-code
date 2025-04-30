import json
from pathlib import Path

import click
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, brier_score_loss
from tira.evaluators import to_prototext


def auc(y_true, y_pred):
    """
    Calculates the ROC-AUC score (Area Under the Curve).

    ROC-AUC is a well-known scalar evaluation score for binary classifiers.

    Parameters
    ----------
    y_true : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`.
    y_pred : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The Area Under the ROC Curve.
    """
    if len(np.unique(y_true)) != 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def c_at_1(y_true, y_pred):
    """
    Calculates the c@1 score.

    This method rewards predictions which leave some problems
    unanswered (score = 0.5). See:
        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    Parameters
    ----------
    y_true : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`, unanswered problems = 0.5.
    y_pred : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The c@1 measure.
    """

    nu = y_pred == 0.5
    nc = np.sum((y_true == (y_pred > 0.5))[~nu])
    nu = np.sum(nu)
    return float((1 / len(y_true)) * (nc + (nu * nc / len(y_true))))


def f1(y_true, y_pred):
    """
    Calculates the F1 score.

    Assesses verification performance, assuming that every score > 0.5
    represents a same-author pair decision.

    Parameters
    ----------
    y_true : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`.
    y_pred : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The F1 score.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """

    score = f1_score(y_true, y_pred > 0.5, zero_division=np.nan)
    if np.isnan(score):
        return None
    return float(score)


def f05u(y_true, y_pred):
    """
    Calculates the F0.5u score.

    F0.5u is a modified F0.5 measure which treats unanswered problems as false negatives.
    as false negatives. See:
        J. Bevendorff et al. Generalizing unmasking for short texts.
        In Proc. of the 14th Conference of the North American Chapter of
        the Association for Computational Linguistics (NAACL 2019). Pages 654–659.

    Parameters
    ----------
    y_true : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`, unanswered problems = 0.5
    y_pred : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    The F0.5u score.
    """

    n_tp = np.sum(y_true * (y_pred > 0.5))
    n_fn = np.sum(y_true * (y_pred < 0.5))
    n_fp = np.sum((1.0 - y_true) * (y_pred > 0.5))
    n_u = np.sum(y_pred == 0.5)

    denom = 1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp
    if denom == 0.0:
        return None
    return float((1.25 * n_tp) / denom)


def brier_score(y_true, y_pred):
    """
    Calculates the complement of the Brier score loss (which is bounded
    to [0-1]), so that higher scores indicate better performance.
    We use the Brier implementation in scikit-learn [Pedregosa et al. 2011].

    Parameters
    ----------
    y_true : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`.
    y_pred : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    Complement of the Brier score.
    """
    return float(1 - brier_score_loss(y_true, np.clip(y_pred, 0.0, 1.0)))


def confusion(y_true, y_pred):
    """
    Calculates the classification confusion matrix.

    Parameters
    ----------
    y_true : array [n_problems]
        The predictions of a verification system.
        Assumes `0 >= prediction <=1`.
    y_pred : array [n_problems]
        The gold annotations provided for each problem as binary labels.

    Returns
    ----------
    Confusion matrix as array.
    """
    return confusion_matrix(y_true, y_pred >= 0.5, labels=[0, 1]).tolist()


def load_problem_file(file_obj):
    problems = {}
    for l in file_obj:
        if not l.strip():
            continue
        j = json.loads(l)
        if 'label' not in j:
            raise ValueError('Invalid input JSON schema')

        d = {k: v for k, v in j.items() if k not in ['id', 'label']}
        if type(j['label']) in [list, tuple]:
            assert j['label'][0] ^ j['label'][1]      # sanity check
            d['label'] = 0.0 if j['label'][0] else 1.0
        elif type(j['label']) in [int, float]:
            d['label'] = float(j['label'])
        else:
            raise ValueError(f'Invalid data type {type(j["label"])} for problem {j["id"]}."')
        problems[j['id']] = d

    return problems


def evaluate_all(y_true, y_pred):
    """
    Calculate all evaluation scores and return results as a dict.
    All scores are rounded to three digits.

    :param y_true : truth as numpy array [n_problems]
    :param y_pred : predictions as numpy array [n_problems]
    """

    results = {
        'roc-auc': auc(y_true, y_pred),
        'brier': brier_score(y_true, y_pred),
        'c@1': c_at_1(y_true, y_pred),
        'f1': f1(y_true, y_pred),
        'f05u': f05u(y_true, y_pred),
    }
    results['mean'] = float(np.mean([v or 0.0 for v in results.values()]))

    for k, v in results.items():
        results[k] = round(v, 3) if v is not None else v

    results['confusion'] = confusion(y_true, y_pred)
    return results


def vectorize(truth_dict, pred_dict, missing_default=0.5):
    """
    Vectorize input data into numpy arrays.

    :param truth_dict: ground truth dict
    :param pred_dict: predictions dict
    :param missing_default: default missing predictions to this value
    :return: truth and predictions numpy arrays
    """
    missing_default = {'label': missing_default}
    scores = [(truth_dict[k]['label'], pred_dict.get(k, missing_default)['label']) for k in sorted(truth_dict)]
    truth, pred = zip(*scores)
    truth = np.array(truth, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)
    assert len(truth) == len(pred)
    return truth, pred


def optimize_pred_scores(y_true, y_pred, max_steps=1000) -> float:
    """
    Optimize c@1 score by finding optimal operating point.

    :param y_true: ground truth array
    :param y_pred: predictions array
    :param max_steps: maximum number of iterations
    :return: optimal operating point
    """

    min_offset = -(np.max(y_pred) - .49)
    max_offset = .51 - np.min(y_pred)
    best_offset = 0.0
    best_score = c_at_1(y_true, y_pred)

    score_updated = False
    for offset in np.linspace(min_offset, max_offset, max_steps):
        score = c_at_1(y_true, y_pred + offset)
        if score > best_score:
            best_score = score
            best_offset = offset
            score_updated = True
        elif score_updated and score < best_score - .01:
            break
    return best_offset


@click.command()
@click.argument('answer_file', type=click.File('r'))
@click.argument('truth_file', type=click.File('r'))
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--outfile-name', default='evaluation.json', show_default=True,
              help='Output JSON filename')
@click.option('-p', '--skip-prototext', is_flag=True,
              help='Skip Tira Prototext output, only write JSON')
@click.option('-s', '--skip-source-eval', is_flag=True, help='Skip evaluation of individual sources')
@click.option('--optimize-score', is_flag=True, help='Optimize score by finding optimal operating point')
def main(answer_file, truth_file, output_dir, outfile_name, skip_prototext, skip_source_eval, optimize_score):
    """
    PAN'25 Generative AI Authorship Verification evaluator.
    """
    pred = load_problem_file(answer_file)
    truth = load_problem_file(truth_file)
    output_dir = Path(output_dir)
    outfile_name = Path(outfile_name)

    click.echo(f'-> {len(truth)} problems in ground truth', err=True)
    click.echo(f'-> {len(pred)} solutions explicitly proposed', err=True)

    if len(pred) > len(truth) or set(truth.keys()).union(set(pred)) != set(truth.keys()):
        raise click.UsageError('Truth file does not match answer file.')

    y_true, y_pred = vectorize(truth, pred)

    # Find optimal operating point if --optimize-score is set
    score_offset = 0.0
    if optimize_score:
        score_offset = optimize_pred_scores(y_true, y_pred)
        y_pred += score_offset
        for k in pred:
            pred[k] = {k_: v_ + score_offset if k_ == 'label' else v_ for k_, v_ in pred[k].items()}

    # Evaluate all test cases
    results = evaluate_all(y_true, y_pred)

    # Write Tira Prototext
    if not skip_prototext:
        with (output_dir / (outfile_name.stem + '.prototext')).open('w') as f:
            f.write(to_prototext([{k: v} for k, v in results.items() if type(v) is float]))

    # Evaluate test cases for individual sources and add to JSON output
    if not skip_source_eval:
        for s in ['source', 'model', 'genre']:
            keys = sorted({v[s] for v in truth.values() if s in v})
            if not keys:
                continue
            results[f'_eval-{s}'] = {k: evaluate_all(*vectorize(
                {k_: v_ for k_, v_ in truth.items() if v_[s] == k}, pred)) for k in keys}

    jstr = json.dumps(results, indent=4, sort_keys=False)
    click.echo(jstr)
    with (output_dir / outfile_name).open('w') as f:
        f.write(jstr)

    if optimize_score:
        click.echo(f'-> Score offset applied: {score_offset}', err=True)


if __name__ == '__main__':
    main()
