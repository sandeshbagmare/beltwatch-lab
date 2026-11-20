"""Keep the checked-in Pages demo consistent with the reference experiment."""

import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from test_report import ReplayHTML

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class SiteHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.rows = {}
        self.row = None
        self.cells = []
        self.cell = None
        self.frame = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag in {"a", "link", "iframe", "img"}:
            target = attributes.get("href") or attributes.get("src")
            if target:
                self.links.append(target)
        if tag == "iframe":
            self.frame = attributes
        if tag == "tr":
            self.row = attributes.get("data-method")
            self.cells = []
        if self.row and tag == "td":
            self.cell = ""

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.cells.append(self.cell.strip())
            self.cell = None
        if tag == "tr" and self.row:
            self.rows[self.row] = self.cells
            self.row = None


def test_pages_links_and_table_match_the_reference_run():
    site = SiteHTML()
    site.feed((DOCS / "index.html").read_text(encoding="utf-8"))
    assert (DOCS / ".nojekyll").exists()
    assert site.frame["src"] == "assets/replay.html"
    assert site.frame["title"] and site.frame["sandbox"] == "allow-scripts"
    for target in site.links:
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        path = (DOCS / unquote(url.path)).resolve() if url.path else DOCS / "index.html"
        assert path.is_relative_to(DOCS)
        assert path.exists(), target
        if url.fragment and path == DOCS / "index.html":
            assert url.fragment in site.ids, target

    reference = json.loads((DOCS / "assets/metrics.json").read_text(encoding="utf-8"))
    assert set(site.rows) == set(reference["methods"])
    for method, scores in reference["methods"].items():
        assert site.rows[method] == [
            f"{scores['clip_auroc']:.3f}",
            f"{scores['frame_ap']:.3f}",
            f"{scores['recall']:.1%}",
            f"{scores['false_alarm_rate']:.1%}",
        ]


def test_published_replay_has_every_method_and_a_matching_script_policy():
    html = (DOCS / "assets/replay.html").read_text(encoding="utf-8")
    parser = ReplayHTML()
    parser.feed(html)
    run = json.loads(parser.payload)
    reference = json.loads((DOCS / "assets/metrics.json").read_text(encoding="utf-8"))
    assert run["methods"] == reference["methods"]
    assert run["config"] == reference["config"]
    for clip in run["clips"]:
        assert set(clip["method_scores"]) == set(reference["methods"])
        assert clip["scores"] == clip["method_scores"]["latent_sigreg"]
        for scores in clip["method_scores"].values():
            assert scores[:4] == [None] * 4
            assert len(scores) == len(clip["labels"])
    [script] = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"'sha256-{digest}'" in html
