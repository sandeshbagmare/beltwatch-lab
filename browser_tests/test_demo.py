"""Exercise the static project site and saved replay in a real local browser.

Run explicitly with: python -m pytest browser_tests
Install the browser extra and `python -m playwright install chromium` first.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Browser, BrowserContext, Frame, Page, expect, sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"
PREFIX = "/beltwatch-lab/"


class ProjectSite(SimpleHTTPRequestHandler):
    """Serve docs at the same project prefix used by GitHub Pages."""

    def do_GET(self) -> None:
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if not self.path.startswith(PREFIX):
            self.send_error(404)
            return
        self.path = "/" + self.path[len(PREFIX) :]
        super().do_GET()

    def log_message(self, _format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def site() -> Iterator[str]:
    handler = partial(ProjectSite, directory=str(DOCS))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{PREFIX}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    executable = os.environ.get("BELTWATCH_BROWSER_EXECUTABLE")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def context(browser: Browser, site: str) -> Iterator[BrowserContext]:
    context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
    errors: list[str] = []
    external: list[str] = []
    origin = urlsplit(site).netloc

    def observe(page: Page) -> None:
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on(
            "request",
            lambda request: external.append(request.url) if urlsplit(request.url).netloc != origin else None,
        )

    context.on("page", observe)
    try:
        yield context
        assert errors == [], f"Browser or CSP errors: {errors}"
        assert external == [], f"Unexpected external requests: {external}"
    finally:
        context.close()


def open_demo(context: BrowserContext, site: str) -> tuple[Page, Frame]:
    page = context.new_page()
    response = page.goto(site, wait_until="networkidle")
    assert response is not None and response.status == 200
    page.get_by_title("Interactive conveyor replay").scroll_into_view_if_needed()
    page.frame_locator("iframe").get_by_label("Detector", exact=True).wait_for()
    replay = next(frame for frame in page.frames if frame.url == site + "assets/replay.html")
    wait_for_plot_resize(replay)
    return page, replay


def wait_for_plot_resize(replay: Frame) -> None:
    replay.wait_for_function(
        """() => {
            const canvas = document.querySelector("#chart"), box = canvas.getBoundingClientRect();
            return box.width > 0 && box.height > 0
                && Math.abs(canvas.width - Math.round(box.width * devicePixelRatio)) <= 1
                && Math.abs(canvas.height - Math.round(box.height * devicePixelRatio)) <= 1;
        }"""
    )


def scientific(value: float) -> str:
    return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")


def test_every_detector_matches_recorded_scores_and_target_frames(context: BrowserContext, site: str) -> None:
    _page, replay = open_demo(context, site)
    data = json.loads(replay.locator("#run-data").inner_text())
    detector = replay.get_by_label("Detector", exact=True)
    picker = replay.get_by_label("Test clip", exact=True)
    timeline = replay.get_by_role("slider")
    assert detector.locator("option").count() == len(data["methods"]) == 5
    assert picker.locator("option").count() == len(data["clips"]) == 4

    for index, clip in enumerate(data["clips"]):
        picker.select_option(str(index))
        for method, scores in clip["method_scores"].items():
            detector.select_option(method)
            threshold = data["methods"][method]["threshold"]
            rendered_metrics = replay.locator("#metrics > div").evaluate_all(
                "nodes => Object.fromEntries(nodes.map(node => [node.querySelector('dt').textContent,"
                " node.querySelector('dd').textContent]))"
            )
            for key, label in (
                ("clip_ap", "Held-out clip AP"),
                ("clip_auroc", "Clip AUROC"),
                ("frame_ap", "Frame AP"),
                ("recall", "Event recall"),
                ("false_alarm_rate", "Normal clips alerted"),
            ):
                assert rendered_metrics[label] == f"{data['methods'][method][key]:.3f}"
            for frame in (0, 3, 4, 11, 20, 31):
                timeline.fill(str(frame))
                score = scores[frame]
                decision = "Context" if score is None else "Alert" if score > threshold else "No alert"
                expect(replay.locator("#decision")).to_have_text(decision)
                event = f"Simulated {clip['fault']}" if clip["labels"][frame] else "No simulated event"
                expect(replay.locator("#event")).to_have_text(event)
                if score is not None:
                    expect(replay.locator("#score")).to_have_text(
                        f"Error {scientific(score)} / threshold {scientific(threshold)}"
                    )
                else:
                    expect(replay.locator("#score")).to_have_text("Gathering 4 context frames")


def test_keyboard_playback_stops_when_the_detector_or_clip_changes(context: BrowserContext, site: str) -> None:
    _page, replay = open_demo(context, site)
    timeline = replay.get_by_role("slider")
    timeline.focus()
    timeline.press("Home")
    for _ in range(3):
        timeline.press("ArrowRight")
    expect(replay.locator("#decision")).to_have_text("Context")
    timeline.press("ArrowRight")
    expect(replay.locator("#decision")).not_to_have_text("Context")

    replay.get_by_role("button", name="Play", exact=True).click()
    replay.wait_for_function("() => Number(document.querySelector('#frame').value) > 4")
    replay.get_by_label("Detector", exact=True).select_option("pixel_shift")
    expect(replay.get_by_role("button", name="Play", exact=True)).to_be_visible()
    paused_at = timeline.input_value()
    replay.wait_for_timeout(250)
    assert timeline.input_value() == paused_at

    timeline.press("End")
    expect(timeline).to_have_value("31")
    replay.get_by_role("button", name="Play", exact=True).click()
    replay.wait_for_function("() => Number(document.querySelector('#frame').value) < 31")
    replay.get_by_label("Test clip", exact=True).select_option("2")
    expect(timeline).to_have_value("0")
    expect(replay.get_by_role("button", name="Play", exact=True)).to_be_visible()


@pytest.mark.parametrize("width", (320, 390, 720, 1440))
def test_page_and_embedded_replay_reflow(context: BrowserContext, site: str, width: int) -> None:
    page, replay = open_demo(context, site)
    page.set_viewport_size({"width": width, "height": 900})
    wait_for_plot_resize(replay)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert replay.evaluate("document.documentElement.scrollWidth <= innerWidth")
    expect(replay.get_by_label("Detector", exact=True)).to_be_visible()
    axis_font_size = replay.locator("#chart").evaluate(
        """canvas => {
            const context = canvas.getContext("2d");
            const pixels = Number(context.font.match(/([0-9.]+)px/)[1]);
            return pixels * context.getTransform().a * canvas.getBoundingClientRect().width / canvas.width;
        }"""
    )
    assert axis_font_size >= 11, f"Chart labels shrink to {axis_font_size:.2f} CSS pixels"

    page.locator(".skip-link").focus()
    page.keyboard.press("Enter")
    assert page.url.endswith("#main")
    page.keyboard.press("Tab")
    expect(page.get_by_role("link", name="Explore the replay")).to_be_focused()


def test_enlarged_text_remains_readable(context: BrowserContext, site: str) -> None:
    page, replay = open_demo(context, site)
    page.evaluate("document.documentElement.style.fontSize = '34px'")
    replay.evaluate("document.documentElement.style.fontSize = '32px'")
    wait_for_plot_resize(replay)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert replay.evaluate("document.documentElement.scrollWidth <= innerWidth")
    expect(replay.get_by_role("slider")).to_be_visible()


def test_chart_uses_sharp_legible_labels_on_high_density_screens(browser: Browser, site: str) -> None:
    context = browser.new_context(viewport={"width": 320, "height": 900}, device_scale_factor=2)
    try:
        _page, replay = open_demo(context, site)
        replay.get_by_role("slider").fill("21")
        chart = replay.locator("#chart").evaluate(
            """canvas => {
                const context = canvas.getContext("2d");
                const width = canvas.getBoundingClientRect().width;
                const font = Number(context.font.match(/([0-9.]+)px/)[1]);
                return {width, buffer: canvas.width, font: font * context.getTransform().a * width / canvas.width};
            }"""
        )
        assert abs(chart["buffer"] - chart["width"] * 2) <= 1
        assert chart["font"] >= 11
    finally:
        context.close()


def test_saved_replay_download_runs_without_network(
    browser: Browser, context: BrowserContext, site: str, tmp_path: Path
) -> None:
    page, _replay = open_demo(context, site)
    with page.expect_download() as pending:
        page.get_by_role("link", name="Save the replay").click()
    download = pending.value
    assert download.suggested_filename == "beltwatch-replay.html"
    target = tmp_path / download.suggested_filename
    download.save_as(target)

    offline = browser.new_context(offline=True)
    errors: list[str] = []
    requests: list[str] = []
    try:
        saved = offline.new_page()
        saved.on("pageerror", lambda error: errors.append(str(error)))
        saved.on("request", lambda request: requests.append(request.url))
        saved.goto(target.as_uri())
        expect(saved.get_by_label("Detector", exact=True)).to_be_visible()
        saved.get_by_label("Test clip", exact=True).select_option("2")
        saved.get_by_label("Detector", exact=True).select_option("pixel_shift")
        saved.get_by_role("slider").fill("20")
        expect(saved.locator("#decision")).to_have_text("Alert")
        assert errors == []
        assert requests == [target.as_uri()]
    finally:
        offline.close()
