# The synthetic conveyor dataset

Every clip in BeltWatch is generated from a numeric seed. The renderer shows parcels moving on a textured belt beneath a fixed camera. A clip has 32 grayscale frames, each 48 pixels wide and 32 pixels high, stored as a `uint8` array. A Boolean label records whether the simulated fault is active in each frame.

Scenes vary in parcel layout, belt texture, starting position, speed, lighting, and sensor noise. The generator separates the scene identities used for training, calibration, and testing.

| Split | Clips per scene | Use |
| --- | --- | --- |
| Train | 1 normal | Learn normal motion. |
| Calibration | 1 normal | Set alert thresholds without seeing faults. |
| Test | 1 normal and 3 fault variants | Measure behavior on new scenes. |

The reference run uses 160 training scenes, 72 calibration scenes, and 80 test scenes. Each test scene has a matched normal clip. Every fault variant is pixel identical to that clip until its event begins, including the noise. After the event begins, the variants retain the same lighting and noise draws so those conditions cannot reveal which fault was chosen.

The first changed frame is also the first positive label. Events start between frame 11 and frame 20, counting from zero, and remain labeled through the end.

- In a **jam**, the belt and parcels stop while sensor noise continues.
- In a **reversal**, the belt and parcels begin moving backward.
- For **debris**, a small stationary patch appears on the belt while the rest keeps moving.

To reproduce the reference scenes with `beltwatch.data.make_split`, use seed `2026` for training, `1073743850` for calibration, and `2147485674` for testing. The command records these values in `metrics.json`. Requesting more scenes with the same seed preserves the scenes already generated. Numeric scene IDs are local to a seed. Keep all variants of a scene together if you create another split.

This generator makes exact labels and paired comparisons easy to inspect. Its simple, constant horizontal motion also favors the fixed-shift baseline. It does not model parcel interactions, camera changes, wear, or the appearance of a real factory. Results describe these generated clips.
