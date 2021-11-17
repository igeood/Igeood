import numpy as np
import sklearn.metrics as skm

import utils.file_manager as fm
from utils.logger import logger

RECALL_LEVEL = 0.95


def compute_metrics(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    recall_level: float = RECALL_LEVEL,
    fpr_only=False,
    print_thr=False,
):
    """Compute evaluation metrics for a binary detection problem. The label to be detected is `1`.

    Args:
        in_scores (np.ndarray): Score for the correctly predicted samples or in-distribution.
        out_scores (np.ndarray): Score for the wrongly predicted samples or out-of-distribution.
        recall_level (float): recall level for calculating FPR at given TPR.

    Returns:
        Tuple[float, float, float, float]: FPR, AUROC, AUPR, DETECTION
    """
    pos = np.ones(len(in_scores))  # configured to detect in-distribution samples
    neg = np.zeros(len(out_scores))

    y_true = np.concatenate([pos, neg]).reshape(-1)
    y_pred = np.concatenate([in_scores, out_scores]).reshape(-1)

    fprs, tprs, thresholds = skm.roc_curve(y_true, y_pred, pos_label=1)  # fpr: s > thr
    fpr_at_tpr = compute_fpr_tpr(tprs, fprs, recall_level)
    index = np.argmin([abs(recall_level - f) for f in tprs])
    thr = thresholds[index]
    if print_thr:
        print(thr)
    if fpr_only:
        return fpr_at_tpr

    auroc = compute_auroc(tprs, fprs)
    aupr = compute_aupr(y_true, y_pred)
    detection = detection_error(in_scores, out_scores)
    return (fpr_at_tpr, auroc, aupr, detection)


def confusion_matrix(scores, corr_pred, threshold, verbose=True):
    # tn -> ood data that was classified as ood (GOOD)
    # fp -> ood data that was classified as in-distribution  BAD)
    # fn -> in-distribution data that was clasified as ood (TUNABLE ~5%)
    # tp -> in-distribution data that was clasified as in-distribution (GOOD)
    if not verbose:
        logger.setLevel(logger.WARNING)
    binary_detector = scores > threshold
    c_mat = skm.confusion_matrix(corr_pred, binary_detector)
    tn, fp, fn, tp = c_mat.ravel()
    logger.debug(f"Amount of one scores data: {sum(corr_pred)} / {len(corr_pred)}")
    logger.debug(f"Threshold: {threshold}")
    logger.debug(
        f"Amount of one scores data detected: {sum(binary_detector[corr_pred])} / {sum(corr_pred)}"
    )
    logger.debug(
        f"""Confusion matrix:
            label/pred  C.1  C.0
                1:     {tp}, {fn}  | {tp + fn} 
                0:     {fp}, {tn}  | {fp + tn}
    """
    )
    return tn, fp, fn, tp


def compute_fpr_tpr(tprs, fprs, recall_level):
    return np.interp(recall_level, tprs, fprs)


def compute_auroc(tprs, fprs):
    return np.trapz(tprs, fprs)


def compute_aupr(y_true, y_pred):
    return skm.average_precision_score(y_true, y_pred)


def get_measures(_pos, _neg, recall_level=RECALL_LEVEL):
    if (len(_pos.shape) == 2 and _pos.shape[1] > 1) or (
        len(_neg.shape) == 2 and _neg.shape[1] > 1
    ):
        raise ValueError("Scores with wrong dimensions.")
    logger.debug(f"recall level {recall_level}")
    pos = np.array(_pos).reshape((-1, 1)).round(decimals=7)
    neg = np.array(_neg).reshape((-1, 1)).round(decimals=7)

    fpr, auroc, aupr, detection = compute_metrics(pos, neg, recall_level)

    return auroc, aupr, fpr, detection


def detection_error(S1, S2):
    unique = np.unique(S2)
    error = 1.0
    for delta in [*unique, unique.max() + 1]:
        tpr = np.sum(np.sum(S1 < delta)) / float(len(S1))
        error2 = np.sum(np.sum(S2 >= delta)) / float(len(S2))
        error = np.minimum(error, (tpr + error2) / 2.0)
    return error


def false_positive_rate(tn, fp, fn, tp):
    return fp / (fp + tn)


def false_negative_rate(tn, fp, fn, tp):
    return fn / (tp + fn)


def true_negative_rate(tn, fp, fn, tp):
    # specificity, selectivity or true negative rate (TNR)
    return tn / (fp + tn)


def precision(tn, fp, fn, tp):
    # precision or positive predictive value (PPV)
    return tp / (tp + fp + 1e-6)


def recall(tn, fp, fn, tp):
    # sensitivity, recall, hit rate, or true positive rate
    return tp / (tp + fn)


def true_positive_rate(tn, fp, fn, tp):
    return recall(tn, fp, fn, tp)


def negative_predictive_value(tn, fp, fn, tp):
    return tn / (tn + fn)


def f1_score(tn, fp, fn, tp):
    return 2 * tp / (2 * tp + fp + fn)


def accuracy_score(tn, fp, fn, tp):
    return (tp + tn) / (tp + tn + fp + fn)


def error_score(tn, fp, fn, tp):
    return 1 - accuracy_score(tn, fp, fn, tp)


def threat_score(tn, fp, fn, tp):
    return tp / (tp + fn + fp)


def print_metrics_and_info(
    s_in,
    s_out,
    nn_name="",
    in_dataset_name="",
    out_dataset_name="",
    method_name="",
    header=True,
    save_flag=False,
    verbose=2,
):
    if verbose in [2, False]:
        logger.setLevel(logger.WARNING)

    auroc, aupr_in, fpr_at_tpr_in, detection = get_measures(s_in, s_out)
    auroc, aupr_out, fpr_at_tpr_out, detection = get_measures(-s_out, -s_in)

    header_lines = [
        "{:31}{:>22}".format("Neural network architecture:", nn_name),
        "{:31}{:>22}".format("In-distribution dataset:", in_dataset_name),
        "{:31}{:>22}".format("Out-of-distribution dataset:", out_dataset_name),
    ]

    output_lines = [
        "{:>34}{:>19}".format("Method:", method_name),
        "{:21}{:13.2f}%".format("FPR at TPR 95% (In):", fpr_at_tpr_in * 100),
        "{:21}{:13.2f}%".format("FPR at TPR 95% (Out):", fpr_at_tpr_out * 100),
        "{:21}{:13.2f}%".format("Detection error:", detection * 100),
        "{:21}{:13.2f}%".format("AUROC:", auroc * 100),
        "{:21}{:13.2f}%".format("AUPR (In):", aupr_in * 100),
        "{:21}{:13.2f}%".format("AUPR (Out):", aupr_out * 100),
    ]
    if verbose:
        if header:
            for line in header_lines:
                print(line)

        for line in output_lines:
            print(line)

    if save_flag:
        f = fm.make_evaluation_metrics_file(
            nn_name, out_dataset_name, fm.clean_title(method_name)
        )
        fm.write_evaluation_metrics_file(f, header_lines, output_lines)
        f.close()

    return fpr_at_tpr_in, fpr_at_tpr_out, detection, auroc, aupr_in, aupr_out
