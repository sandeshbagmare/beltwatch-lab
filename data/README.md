# Synthetic conveyor data

BeltWatch Lab generates all footage locally from numeric seeds. It ships no camera footage, external dataset or data downloads. Each clip contains 32 grayscale frames at 48 × 32 pixels, stored as `uint8` arrays, and one Boolean label per frame.

The renderer moves parcels and a textured belt past a fixed overhead view. Each scene samples its parcel layout, belt texture, starting position, speed, lighting and pixel noise. The three splits draw from disjoint scene identities for a given benchmark seed:

| Split | Clips per scene | Purpose |
| --- | --- | --- |
| Train | 1 normal | Fit the model to nominal motion |
| Calibration | 1 normal | Set thresholds without observing faults |
| Test | 1 normal and 3 fault variants | Measure detection on unseen scenes |

Each test scene has a paired normal reference. Before the event, every variant is pixel identical to that reference. Each variant keeps the same lighting and per-frame sensor noise throughout the clip. The interval of positive labels runs from the first changed frame through the end of the clip. Event onset varies from frame 11 to 20 (zero based):

- **Jam**: the belt and parcels stop at their most recent positions while sensor noise continues.
- **Reverse**: the belt and parcels immediately begin moving backward.
- **Debris**: a small stationary patch appears in the belt area while the belt and parcels keep moving.

To reproduce the reference scenes with `beltwatch.data.make_split`, use seed `2026` for train, `1073743850` for calibration, and `2147485674` for test. The CLI records these seeds in its metrics file. A scene stays the same when requesting a larger split with the same seed. Keep every variant of a scene together if you build another partition. Numeric scene IDs are local to one seed.

The generator represents three simple visual changes under a fixed view. It does not model real equipment failures, interactions between parcels, changing cameras, or wear over time. Use the benchmark to compare methods under the same controlled conditions.
