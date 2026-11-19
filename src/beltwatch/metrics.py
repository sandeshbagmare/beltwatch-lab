"""Use normal clip peaks for calibration and evaluate against held-out labels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from beltwatch.data import CONTEXT, Clip


class FaultMetrics(TypedDict):
    support: int
    clip_ap: float
    clip_auroc: float
    recall: float
    median_delay_frames: float | None


class MethodMetrics(TypedDict):
    clip_ap: float
    clip_auroc: float
    frame_ap: float
    recall: float
    false_alarm_rate: float
    threshold: float
    per_fault: dict[str, FaultMetrics]


def binary_ranking(labels: ArrayLike, scores: ArrayLike) -> tuple[float, float]:
    """Return average precision and AUROC, grouping all equal scores."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("ranking requires matching 1D labels and finite scores")
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("ranking requires both positive and negative labels")

    order = np.argsort(-s, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(s[order]) != 0), len(s) - 1]
    true_positive = np.cumsum(y[order])[ends]
    false_positive = ends + 1 - true_positive
    recall = true_positive / positives
    false_rate = false_positive / negatives
    ap = np.sum(np.diff(np.r_[0.0, recall]) * true_positive / (ends + 1))
    auroc = np.trapezoid(np.r_[0.0, recall], np.r_[0.0, false_rate])
    return float(ap), float(auroc)


def calibrated_threshold(scores: ArrayLike, *, alpha: float = 0.10) -> float:
    """Split-conformal order statistic of peak scores on independent normal clips."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all() or not 0 < alpha < 1:
        raise ValueError("calibration requires a nonempty finite clip-score matrix and 0 < alpha < 1")
    peaks = np.max(matrix, axis=1)
    rank = math.ceil((len(peaks) + 1) * (1.0 - alpha))
    if rank > len(peaks):
        raise ValueError("calibration has too few normal clips for this alpha")
    return float(np.partition(peaks, rank - 1)[rank - 1])


def evaluate(clips: Sequence[Clip], scores: ArrayLike, *, threshold: float) -> MethodMetrics:
    """Evaluate clip ranking and target-frame alerts at an externally set threshold."""
    matrix = np.asarray(scores, dtype=np.float64)
    labels = np.stack([clip.labels[CONTEXT:] for clip in clips])
    if not labels.size or matrix.shape != labels.shape or not np.isfinite(matrix).all() or not math.isfinite(threshold):
        raise ValueError("scores must be finite and aligned with the labeled target frames")

    clip_labels = labels.any(axis=1)
    peaks = matrix.max(axis=1)
    clip_ap, clip_auroc = binary_ranking(clip_labels, peaks)
    frame_ap, _ = binary_ranking(labels.ravel(), matrix.ravel())
    alerts = matrix > threshold
    event_hits = (alerts & labels).any(axis=1)
    names = np.asarray([clip.fault for clip in clips])
    per_fault: dict[str, FaultMetrics] = {}

    for fault in sorted(set(names[clip_labels])):
        group = (names == fault) & clip_labels
        subset = group | ~clip_labels
        ap, auroc = binary_ranking(clip_labels[subset], peaks[subset])
        delays: NDArray[np.intp] = (alerts & labels)[group & event_hits].argmax(axis=1) - labels[
            group & event_hits
        ].argmax(axis=1)
        per_fault[str(fault)] = {
            "support": int(group.sum()),
            "clip_ap": ap,
            "clip_auroc": auroc,
            "recall": float(event_hits[group].mean()),
            "median_delay_frames": float(np.median(delays)) if delays.size else None,
        }

    return {
        "clip_ap": clip_ap,
        "clip_auroc": clip_auroc,
        "frame_ap": frame_ap,
        "recall": float(event_hits[clip_labels].mean()),
        "false_alarm_rate": float(alerts[~clip_labels].any(axis=1).mean()),
        "threshold": threshold,
        "per_fault": per_fault,
    }
