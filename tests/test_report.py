import base64
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beltwatch.data import make_split
from beltwatch.report import write_report


class ReplayHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.payload = ""
        self.script_types: list[str | None] = []
        self._reading_data = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            attributes = dict(attrs)
            self.script_types.append(attributes.get("type"))
            assert attributes.get("src") is None
            self._reading_data = attributes.get("type") == "application/json"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._reading_data = False

    def handle_data(self, data: str) -> None:
        if self._reading_data:
            self.payload += data


def test_replay_keeps_scores_at_target_frames_and_renders_snapshot(tmp_path: Path) -> None:
    _, jam, *_ = make_split("test", 1, seed=42)
    scores = np.array([0.125, 0.5, *([0.25] * 26)])
    raw_label = '</script><script type="application/json">"example"</script>'

    write_report(
        tmp_path,
        [(jam, scores)],
        {"methods": {"latent_sigreg": {"threshold": 0.25}}, "config": {"label": raw_label}},
    )

    page = ReplayHTML()
    page.feed((tmp_path / "replay.html").read_text(encoding="utf-8"))
    assert page.script_types == ["application/json", None]
    report = json.loads(page.payload)
    [example] = report["clips"]
    assert report["config"]["label"] == raw_label
    assert report["methods"]["latent_sigreg"]["threshold"] == 0.25
    assert example["scores"][:7] == [None, None, None, None, 0.125, 0.5, 0.25]
    assert len(example["scores"]) == len(example["labels"]) == 32
    assert example["labels"] == jam.labels.tolist()
    assert base64.b64decode(example["pixels"]) == jam.frames.tobytes()
    with Image.open(tmp_path / "snapshot.png") as snapshot:
        assert snapshot.format == "PNG"
        assert snapshot.width >= 900 and snapshot.height >= 400
        grayscale = np.asarray(snapshot.convert("L"), dtype=np.int16)
        assert grayscale.max() - grayscale.min() > 100


def test_report_rejects_scores_for_context_frames(tmp_path: Path) -> None:
    clip = make_split("test", 1)[0]
    with pytest.raises(ValueError, match="target frames"):
        write_report(
            tmp_path,
            [(clip, np.zeros(32))],
            {"methods": {"latent_sigreg": {"threshold": 0.25}}, "config": {}},
        )
    assert not list(tmp_path.iterdir())


def test_comparison_scores_align_with_the_same_target_frames(tmp_path: Path) -> None:
    clips = make_split("test", 1)[:2]
    primary = np.full(28, 0.125)
    comparisons = [np.full(28, 0.5), np.full(28, 0.75)]
    write_report(
        tmp_path,
        [(clip, primary) for clip in clips],
        {
            "methods": {"latent_sigreg": {"threshold": 0.25}, "pixel_delta": {"threshold": 0.4}},
            "config": {},
        },
        comparison_scores={"pixel_delta": comparisons},
    )
    page = ReplayHTML()
    page.feed((tmp_path / "replay.html").read_text(encoding="utf-8"))
    report = json.loads(page.payload)
    for clip, comparison in zip(report["clips"], comparisons, strict=True):
        assert clip["method_scores"]["latent_sigreg"] == clip["scores"]
        assert clip["method_scores"]["pixel_delta"] == [None] * 4 + comparison.tolist()


@pytest.mark.parametrize(
    "method, values, threshold",
    [
        ("missing", [np.zeros(28)], 0.25),
        ("pixel_delta", [], 0.25),
        ("pixel_delta", [np.zeros(32)], 0.25),
        ("pixel_delta", [np.full(28, float("nan"))], 0.25),
        ("pixel_delta", [np.full(28, -1)], 0.25),
        ("pixel_delta", [np.zeros(28)], float("nan")),
        ("latent_sigreg", [np.ones(28)], 0.25),
    ],
)
def test_invalid_comparison_never_writes_partial_report(tmp_path, method, values, threshold):
    clip = make_split("test", 1)[0]
    with pytest.raises(ValueError, match="comparison"):
        write_report(
            tmp_path,
            [(clip, np.zeros(28))],
            {
                "methods": {"latent_sigreg": {"threshold": 0.25}, "pixel_delta": {"threshold": threshold}},
                "config": {},
            },
            comparison_scores={method: values},
        )
    assert not list(tmp_path.iterdir())
