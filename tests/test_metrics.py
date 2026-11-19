import math

import numpy as np
import pytest

from beltwatch.data import CONTEXT, Clip
from beltwatch.metrics import binary_ranking, calibrated_threshold, evaluate


def test_equal_scores_have_prevalence_ap_and_half_auroc():
    ap, auroc = binary_ranking([True, False, False, True, False], [1, 1, 1, 1, 1])
    assert ap == pytest.approx(0.4)
    assert auroc == pytest.approx(0.5)


def test_ranking_with_one_tied_group_has_analytic_result():
    ap, auroc = binary_ranking([True, False, True, False], [3, 2, 2, 1])
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert auroc == pytest.approx(0.875)


def test_calibration_operates_on_clip_peaks_with_finite_sample_rank():
    matrix = np.repeat(np.arange(10, dtype=float)[:, None], 3, axis=1)
    assert calibrated_threshold(matrix, alpha=0.2) == 8
    with pytest.raises(ValueError, match="calibration"):
        calibrated_threshold(matrix[:4], alpha=0.1)


def test_evaluation_aligns_to_target_frames_and_uses_strict_alert_threshold():
    frames = np.zeros((CONTEXT + 3, 32, 48), dtype=np.uint8)
    nominal = Clip("n", 1, "normal", frames, np.zeros(len(frames), dtype=bool))
    fault_labels = np.array([False] * CONTEXT + [False, True, True])
    fault = Clip("f", 1, "jam", frames, fault_labels)
    result = evaluate([nominal, fault], np.array([[0.1, 0.4, 0.2], [0.1, 0.8, 1.0]]), threshold=0.4)

    assert result["clip_ap"] == result["clip_auroc"] == result["frame_ap"] == 1.0
    assert result["false_alarm_rate"] == 0.0
    assert result["recall"] == 1.0
    assert result["per_fault"]["jam"]["median_delay_frames"] == 0
    assert math.isfinite(result["threshold"])


def test_early_false_alert_does_not_count_as_a_fault_hit():
    frames = np.zeros((CONTEXT + 3, 32, 48), dtype=np.uint8)
    nominal = Clip("n", 1, "normal", frames, np.zeros(len(frames), dtype=bool))
    fault = Clip("f", 1, "jam", frames, np.array([False] * CONTEXT + [False, True, True]))
    result = evaluate([nominal, fault], np.array([[0.1, 0.2, 0.2], [0.8, 0.1, 0.1]]), threshold=0.4)
    assert result["recall"] == 0.0
    assert result["per_fault"]["jam"]["median_delay_frames"] is None
