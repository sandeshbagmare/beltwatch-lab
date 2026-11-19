from itertools import combinations

import numpy as np
from numpy.testing import assert_array_equal

from beltwatch.data import CONTEXT, FAULTS, make_split


def test_split_is_reproducible_and_scene_prefix_does_not_depend_on_size() -> None:
    first, *_ = make_split("train", 3, seed=62)
    (repeated,) = make_split("train", 1, seed=62)
    (another_seed,) = make_split("train", 1, seed=63)

    assert first.clip_id == repeated.clip_id
    assert first.scene_id == repeated.scene_id
    assert_array_equal(first.frames, repeated.frames)
    assert_array_equal(first.labels, repeated.labels)
    assert not np.array_equal(first.frames, another_seed.frames)


def test_training_and_calibration_are_normal_and_splits_share_no_scenes() -> None:
    train = make_split("train", 4)
    calibration = make_split("calibration", 4)
    test = make_split("test", 4)

    for clip in train + calibration:
        assert clip.fault == "normal"
        assert not clip.labels.any()

    for left, right in combinations((train, calibration, test), 2):
        assert {clip.scene_id for clip in left}.isdisjoint(clip.scene_id for clip in right)
        left_frames = {frame.tobytes() for clip in left for frame in clip.frames}
        assert left_frames.isdisjoint(frame.tobytes() for clip in right for frame in clip.frames)


def test_each_fault_pair_first_changes_on_its_first_labeled_frame() -> None:
    clips = make_split("test", 8)
    scenes = {clip.scene_id for clip in clips}

    assert len(scenes) == 8
    assert len({clip.clip_id for clip in clips}) == len(clips)
    assert {"jam", "reverse", "debris"} <= set(FAULTS)
    onsets = set()
    for scene_id in scenes:
        scene = [clip for clip in clips if clip.scene_id == scene_id]
        assert len(scene) == len(FAULTS) + 1
        assert {clip.fault for clip in scene} == {"normal", *FAULTS}
        normal = next(clip for clip in scene if clip.fault == "normal")
        assert not normal.labels.any()
        for clip in scene:
            assert clip.frames.shape == (32, 32, 48)
            assert clip.frames.dtype == np.uint8
            assert clip.labels.shape == (len(clip.frames),)
            assert clip.labels.dtype == np.bool_
            if clip.fault == "normal":
                continue

            changed = np.flatnonzero(np.any(clip.frames != normal.frames, axis=(1, 2)))
            onset = int(np.flatnonzero(clip.labels)[0])
            assert CONTEXT <= onset < len(clip.frames) - 1
            assert changed[0] == onset
            assert clip.labels[onset:].all()
            onsets.add(onset)

    assert len(onsets) > 1
