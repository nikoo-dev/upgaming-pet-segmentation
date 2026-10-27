"""Rows are target, columns prediction; ignored pixels never enter the matrix."""
import numpy as np


def confusion(prediction, target):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target dimensions differ")
    if not np.isin(target, [0, 1, 255]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Unexpected mask class")
    valid = target != 255
    return np.bincount((2 * target[valid].astype(np.int64) + prediction[valid]).ravel(), minlength=4).reshape(2, 2)


def divide(a, b):
    return float(a / b) if b else None


def scores(matrix):
    m = np.asarray(matrix, dtype=np.int64)
    tn, fp, fn, tp = (int(x) for x in m.ravel())
    fg = divide(tp, tp + fp + fn)
    bg = divide(tn, tn + fp + fn)
    defined = [x for x in [bg, fg] if x is not None]
    return {"foreground_iou": fg, "background_iou": bg,
            "mean_iou": sum(defined) / len(defined) if defined else None,
            "foreground_dice": divide(2 * tp, 2 * tp + fp + fn),
            "pixel_accuracy": divide(tp + tn, tp + tn + fp + fn),
            "confusion_matrix": m.tolist()}
