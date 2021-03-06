from sklearn import metrics

def auc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def highest_tpr_thresh(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return thresholds[tpr.argmax()]


def lowest_fpr_thresh(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return thresholds[fpr.argmin()]


def acc(y_true, y_pred):
    return metrics.accuracy_score(y_true, (y_pred > 0.5) * 1)