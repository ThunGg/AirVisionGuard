import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import math


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidean distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        # Clamp for numerical stability
        similarity = np.clip(similarity, -1.0, 1.0)
        dist = np.arccos(similarity) / math.pi
    else:
        raise ValueError('Undefined distance metric %d' % distance_metric)
    return dist


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, len(thresholds)))
    fprs = np.zeros((nrof_folds, len(thresholds)))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0

        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find best threshold using sorting (Optimization)
        train_dist = dist[train_set]
        train_issame = actual_issame[train_set]

        # Sort training distances
        sorted_idx = np.argsort(train_dist)
        sorted_dist = train_dist[sorted_idx]
        sorted_issame = train_issame[sorted_idx]

        fp_total = int(np.sum(np.logical_not(sorted_issame)))
        tp_total = len(sorted_issame) - fp_total

        # FIX 1a — Correct accuracy formula.
        #
        # The old code built (N+1)-length tp/fp arrays representing the number of
        # same/different pairs ABOVE a threshold ("remaining"), then computed:
        #
        #   acc = (tp_remaining + fp_total - fp_remaining) / N
        #       = (FN + FP) / N   ← error rate, not accuracy
        #
        # argmax of the error rate finds the WORST threshold, not the best.
        #
        # Fix: build cumulative sums from below (accepted pairs) so that
        # cs_issame[i] = TP and cs_diff[i] = FP when threshold = sorted_dist[i].
        # Accuracy = (TP + TN) / N = (cs_issame[i] + fp_total - cs_diff[i]) / N.
        #
        # The length-N+1 arrays cover thresholds sorted_dist[0..N-1] plus one
        # extra slot (index N) representing "accept all pairs".
        cs_issame = np.concatenate([[0], np.cumsum(sorted_issame)])   # cs_issame[i] = TP at sorted_dist[i]
        cs_diff   = np.concatenate([[0], np.cumsum(np.logical_not(sorted_issame))])  # cs_diff[i]   = FP at sorted_dist[i]
        acc = (cs_issame + fp_total - cs_diff) / len(train_set)

        # FIX 1b — Correct threshold index mapping.
        #
        # The old code indexed sorted_dist[best_threshold_index - 1], which is
        # off by one: it maps acc[i] → sorted_dist[i-1], i.e. the distance BELOW
        # the decision boundary, causing it to produce the threshold for the
        # previous step.  Also, when best_threshold_index == 0 it defaulted to
        # 0.0 regardless of the actual minimum distance.
        #
        # Fix: acc[i] was computed assuming threshold = sorted_dist[i], so map
        # directly.  The extra slot at index N (accept all) requires a threshold
        # slightly above the maximum observed distance.
        best_threshold_index = np.argmax(acc)
        if best_threshold_index < len(sorted_dist):
            best_threshold = sorted_dist[best_threshold_index]
        else:
            best_threshold = sorted_dist[-1] + 1e-6

        # Evaluate ROC for fixed thresholds using test set
        test_dist = dist[test_set]
        test_issame = actual_issame[test_set]

        for i, threshold in enumerate(thresholds):
            tprs[fold_idx, i], fprs[fold_idx, i], _ = calculate_accuracy(threshold, test_dist, test_issame)

        _, _, accuracy[fold_idx] = calculate_accuracy(best_threshold, test_dist, test_issame)

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # NOTE: the `thresholds` parameter is not used in this function; the
    # threshold is derived from far_target instead.  It is kept for backward
    # compatibility with existing call sites.
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0

        dist_all = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)
        train_dist   = dist_all[train_set]
        train_issame = actual_issame[train_set]

        diff_dist = train_dist[np.logical_not(train_issame)]

        # FIX 2 — Correct FAR threshold derivation.
        #
        # The old code used:
        #   target_idx = int(nrof_diff * far_target)
        #   threshold  = sorted_diff_dist[target_idx]
        #
        # For small folds (e.g. nrof_diff=500, far_target=1e-3) this truncates
        # to target_idx=0 and sets the threshold to the minimum impostor distance,
        # so nothing is ever accepted and VAL ≈ 0.
        #
        # Fix: use np.percentile to find the far_target-th quantile of the
        # impostor distance distribution. This interpolates correctly for small
        # sample sizes and handles the edge case gracefully.
        threshold = np.percentile(diff_dist, far_target * 100)

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist_all[test_set], actual_issame[test_set])

    return np.mean(val), np.std(val), np.mean(far)


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept  = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept)  / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    actual_issame = np.asarray(actual_issame).astype(bool)
    # FIX 3 — Adapt threshold range to the distance metric.
    #
    # The old code always used np.arange(0, 4, 0.01). Cosine angular distance
    # (metric=1) is bounded to [0, 1] by construction (arccos / π), so
    # thresholds above 1.0 are redundant and waste computation.  Squared
    # Euclidean distance (metric=0) can legitimately reach ~4 for unit-normalised
    # embeddings, so [0, 4) with step 0.01 is kept unchanged.
    if distance_metric == 1:
        thresholds = np.arange(0, 1.0 + 1e-9, 0.001)  # finer step, correct range
    else:
        thresholds = np.arange(0, 4, 0.01)

    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy = calculate_roc(
        thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        distance_metric=distance_metric,
        subtract_mean=subtract_mean,
    )

    val, val_std, far = calculate_val(
        None, embeddings1, embeddings2,
        np.asarray(actual_issame),
        1e-3,
        nrof_folds=nrof_folds,
        distance_metric=distance_metric,
        subtract_mean=subtract_mean,
    )

    return tpr, fpr, accuracy, val, val_std, far