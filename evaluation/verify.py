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
        
        tp_total = np.sum(sorted_issame)
        fp_total = len(sorted_issame) - tp_total
        
        # Vectorized TP/FP/TN/FN for all possible thresholds in training
        tp = tp_total - np.cumsum(sorted_issame)
        fp = fp_total - np.cumsum(~sorted_issame)
        # Add boundary case (threshold lower than all distances)
        tp = np.concatenate([[tp_total], tp])
        fp = np.concatenate([[fp_total], fp])
        
        acc = (tp + (fp_total - fp)) / len(train_set)
        best_threshold_index = np.argmax(acc)
        best_threshold = sorted_dist[best_threshold_index-1] if best_threshold_index > 0 else 0.0
        
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
        train_dist = dist_all[train_set]
        train_issame = actual_issame[train_set]
        
        # Sort training distances to find threshold for FAR (Optimization)
        diff_dist = train_dist[~train_issame]
        sorted_diff_dist = np.sort(diff_dist)
        
        # The threshold for a given FAR is the n-th quantile of different-pair distances
        nrof_diff = len(diff_dist)
        target_idx = int(nrof_diff * far_target)
        
        if target_idx < nrof_diff:
            threshold = sorted_diff_dist[target_idx]
        else:
            threshold = sorted_diff_dist[-1]

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist_all[test_set], actual_issame[test_set])

    return np.mean(val), np.std(val), np.mean(far)

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Use standard threshold range for ROC return consistency
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    # Val at FAR calculation
    val, val_std, far = calculate_val(None, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    return tpr, fpr, accuracy, val, val_std, far
