"""Generate paired normal and faulty conveyor clips without external data."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CONTEXT = 4
FAULTS: tuple[str, ...] = ("jam", "reverse", "debris")

_SPLITS = ("train", "calibration", "test")
_LENGTH, _HEIGHT, _WIDTH = 32, 32, 48
_BELT = slice(10, 22)
_BELT_HEIGHT = _BELT.stop - _BELT.start
_LOOP = 78
_PERIOD = 13


@dataclass(frozen=True)
class Clip:
    """One generated video with a Boolean event label per observed frame."""

    clip_id: str
    scene_id: int
    fault: str
    frames: NDArray[np.uint8]
    labels: NDArray[np.bool_]


def make_split(split: Literal["train", "calibration", "test"], n_scenes: int, *, seed: int = 2026) -> list[Clip]:
    """Generate independent scenes; only the test split includes fault variants."""
    split_index = _SPLITS.index(split)
    if n_scenes < 0 or seed < 0:
        raise ValueError("n_scenes and seed must be nonnegative")

    variants = ("normal", *FAULTS) if split == "test" else ("normal",)
    clips = []
    for index in range(n_scenes):
        # Interleaving leaves each split an unbounded, disjoint ID range.
        scene_id = index * len(_SPLITS) + split_index
        rng = np.random.default_rng(np.random.SeedSequence((seed, scene_id)))
        for fault, frames, labels in _render_scene(rng, variants):
            clip_id = f"{split}-{seed}-{scene_id:04d}-{fault}"
            clips.append(Clip(clip_id, scene_id, fault, frames, labels))
    return clips


def _render_scene(
    rng: np.random.Generator, variants: tuple[str, ...]
) -> list[tuple[str, NDArray[np.uint8], NDArray[np.bool_]]]:
    background = np.full((_HEIGHT, _WIDTH), 35, dtype=np.int16)
    background[4:28] = 54
    background[6:8] = background[24:26] = 146
    background[[6, 25], 4::9] = 195
    background[[9, 22]] = 72
    texture = rng.integers(93, 107, size=(_BELT_HEIGHT, _PERIOD), dtype=np.int16)
    texture[:, 0] -= 22
    texture[[0, -1]] -= 13

    # Unroll the moving parcels along a loop wider than the camera crop.
    parcels = np.zeros((_BELT_HEIGHT, _LOOP), dtype=np.int16)
    for slot in range(0, _LOOP, 26):
        top = int(rng.integers(1, 5))
        left = slot + int(rng.integers(0, 12))
        height, width = int(rng.integers(5, 8)), int(rng.integers(6, 13))
        box = parcels[top : top + height, left : left + width]
        box[:] = int(rng.integers(151, 203))
        box[:, [0, -1]] -= 27
        box[[0, -1]] -= 19
        box[1:-1, width // 2] += 21

    frame_numbers = np.arange(_LENGTH)
    onset = int(rng.integers(11, 21))
    start, speed = int(rng.integers(0, _LOOP)), int(rng.integers(1, 3))
    nominal = start + frame_numbers * speed
    columns = np.arange(_WIDTH)
    debris_top, debris_left = int(rng.integers(13, 18)), int(rng.integers(5, 39))
    debris = np.array([[57, 64, 67, 68], [56, 85, 81, 60], [50, 50, 54, 48]])
    # Reuse the camera conditions across variants so they cannot reveal the event by themselves.
    light = int(rng.integers(-12, 13)) + np.rint(np.linspace(-7, 7, _WIDTH)).astype(np.int16)
    noise = rng.integers(-3, 4, size=(_LENGTH, _HEIGHT, _WIDTH), dtype=np.int16)

    results = []
    for fault in variants:
        positions = nominal.copy()
        if fault == "jam":
            positions[onset:] = nominal[onset - 1]
        elif fault == "reverse":
            reverse_steps = frame_numbers[onset:] - (onset - 1)
            positions[onset:] = nominal[onset - 1] - speed * reverse_steps

        raw = np.broadcast_to(background, noise.shape).copy()
        for index, position in enumerate(positions):
            coordinates = columns - position
            cargo = parcels[:, coordinates % _LOOP]
            raw[index, _BELT] = np.where(cargo, cargo, texture[:, coordinates % _PERIOD])
        if fault == "debris":
            raw[onset:, debris_top : debris_top + 3, debris_left : debris_left + 4] = debris
        frames = np.clip(raw + noise + light, 0, 255).astype(np.uint8)
        labels = (frame_numbers >= onset) & (fault != "normal")
        results.append((fault, frames, labels))
    return results
