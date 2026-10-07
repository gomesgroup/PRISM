"""Scoring, kept in one place so every table means the same thing."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def accuracy(truth: Sequence[int], predicted: Sequence[int]) -> float:
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    return float((truth == predicted).mean()) if len(truth) else float("nan")


def r2(truth: Sequence[float], predicted: Sequence[float]) -> float:
    """Coefficient of determination; NaN when the truth has no spread."""
    truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
    if len(truth) < 3:
        return float("nan")
    sst = float(((truth - truth.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - ((predicted - truth) ** 2).sum() / sst)


def mae(truth: Sequence[float], predicted: Sequence[float]) -> float:
    truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
    return float(np.abs(predicted - truth).mean()) if len(truth) else float("nan")


def confusion(truth: Sequence[int], predicted: Sequence[int]) -> Dict[str, int]:
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    return {"tp": int(((truth == 1) & (predicted == 1)).sum()),
            "fp": int(((truth == 0) & (predicted == 1)).sum()),
            "fn": int(((truth == 1) & (predicted == 0)).sum()),
            "tn": int(((truth == 0) & (predicted == 0)).sum())}


def classification_report(truth: Sequence[int], predicted: Sequence[int]) -> Dict[str, float]:
    c = confusion(truth, predicted)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    truth = np.asarray(truth)
    majority = float(max(truth.mean(), 1 - truth.mean())) if len(truth) else float("nan")
    return {**c, "n": int(len(truth)), "accuracy": accuracy(truth, predicted),
            "majority": majority,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def joint_score(class_accuracy: float, rate_r2: float,
                class_weight: float = 0.5) -> float:
    """The single number used to rank candidates.

    One model serves both tasks, so selection needs one criterion. R2 is
    floored at zero so that a candidate cannot be rewarded for a negative fit
    being "less negative" than another's.
    """
    return class_weight * class_accuracy + (1 - class_weight) * max(rate_r2, 0.0)
