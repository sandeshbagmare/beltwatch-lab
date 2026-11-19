"""Render the held-out replay without a server or browser dependencies."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from beltwatch.data import CONTEXT, Clip

_Example = tuple[Clip, NDArray[np.float64]]

_BACKGROUND = "#0a111d"
_PANEL = "#131f30"
_GRID = "#31445c"
_TEXT = "#edf5ff"
_MUTED = "#a1b4cc"
_SCORE = "#73ecd8"
_FAULT = "#ffbb78"
_FAULT_BG = "#302a2c"


def write_report(directory: Path, examples: Sequence[_Example], metrics: Mapping[str, Any]) -> None:
    """Write a standalone replay and a figure using the same scores and event labels."""
    threshold = float(metrics["methods"]["latent_sigreg"]["threshold"])
    if not examples or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("report requires examples and a finite, nonnegative threshold")

    clips = []
    for clip, scores in examples:
        length, height, width = clip.frames.shape
        if (
            length <= CONTEXT
            or clip.labels.shape != (length,)
            or scores.shape != (length - CONTEXT,)
            or not np.isfinite(scores).all()
            or (scores < 0).any()
        ):
            raise ValueError("report scores must be finite, nonnegative and aligned to target frames")
        clips.append(
            {
                "id": clip.clip_id,
                "scene": clip.scene_id,
                "fault": clip.fault,
                "width": width,
                "height": height,
                "pixels": base64.b64encode(clip.frames.tobytes()).decode("ascii"),
                "labels": clip.labels.tolist(),
                "scores": [None] * CONTEXT + scores.tolist(),
            }
        )

    payload = json.dumps(
        {"clips": clips, "context": CONTEXT, "methods": metrics["methods"], "config": metrics["config"]},
        allow_nan=False,
        separators=(",", ":"),
    ).replace("<", r"\u003c")  # The HTML parser still recognizes end tags inside an inert JSON script.
    script_hash = base64.b64encode(hashlib.sha256(_SCRIPT.encode("utf-8")).digest()).decode("ascii")
    html = _HTML.replace("@HASH@", script_hash).replace("@SCRIPT@", _SCRIPT).replace("@DATA@", payload)
    (directory / "replay.html").write_bytes(html.encode("utf-8"))
    _snapshot(examples, metrics, threshold).save(directory / "snapshot.png")


def _snapshot(examples: Sequence[_Example], metrics: Mapping[str, Any], threshold: float) -> Image.Image:
    rows = (len(examples) + 1) // 2
    height = max(430, 234 + rows * 212)
    image = Image.new("RGB", (1200, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=43)
    subtitle = ImageFont.load_default(size=23)
    small = ImageFont.load_default(size=16)
    tiny = ImageFont.load_default(size=14)

    draw.text((40, 34), "BELTWATCH LAB    /    HELD-OUT REPLAY", font=small, fill=_SCORE)
    draw.text((38, 65), "A conveyor, seen through its prediction error", font=title, fill=_TEXT)
    draw.text(
        (40, 133),
        "Regularized latent model  /  Synthetic footage  /  Examples share the same score scale",
        font=small,
        fill=_MUTED,
    )

    high = max(threshold, *(float(scores.max()) for _, scores in examples)) or 1.0
    for index, (clip, scores) in enumerate(examples):
        panel_x, panel_y = 40 + 568 * (index % 2), 177 + 212 * (index // 2)
        draw.rounded_rectangle((panel_x, panel_y, panel_x + 552, panel_y + 196), radius=13, fill=_PANEL, outline=_GRID)
        draw.text((panel_x + 18, panel_y + 14), clip.fault.capitalize(), font=subtitle, fill=_TEXT)
        draw.text((panel_x + 18, panel_y + 46), f"Scene {clip.scene_id:04}", font=tiny, fill=_MUTED)
        onsets = np.flatnonzero(clip.labels)
        onset = int(onsets[0]) if len(onsets) else None
        start = max(CONTEXT, onset if onset is not None else CONTEXT)
        selected = start + int(scores[start - CONTEXT :].argmax())
        frame = Image.fromarray(clip.frames[selected]).resize((144, 96), Image.Resampling.NEAREST)
        image.paste(frame, (panel_x + 18, panel_y + 69))
        draw.text((panel_x + 18, panel_y + 170), f"Frame {selected:02}", font=tiny, fill=_MUTED)

        left, top, right, bottom = panel_x + 185, panel_y + 56, panel_x + 532, panel_y + 160
        plot_width, plot_height = right - left, bottom - top
        last = len(clip.frames) - 1
        if onset is not None:
            draw.rectangle((left + plot_width * onset / last, top, right, bottom), fill=_FAULT_BG)
        draw.line((left, bottom, right, bottom), fill=_GRID)
        limit_y = bottom - plot_height * threshold / high
        for x in range(left, right, 12):
            draw.line((x, limit_y, min(x + 6, right), limit_y), fill=_FAULT, width=2)
        for previous, score in enumerate(scores[1:], start=CONTEXT):
            draw.line(
                (
                    left + plot_width * previous / last,
                    bottom - plot_height * float(scores[previous - CONTEXT]) / high,
                    left + plot_width * (previous + 1) / last,
                    bottom - plot_height * float(score) / high,
                ),
                fill=_SCORE,
                width=2,
            )
        cursor_x = left + plot_width * selected / last
        cursor_y = bottom - plot_height * float(scores[selected - CONTEXT]) / high
        draw.ellipse((cursor_x - 4, cursor_y - 4, cursor_x + 4, cursor_y + 4), fill=_SCORE)
        draw.text((left, panel_y + 170), "Frame 00", font=tiny, fill=_MUTED)
        draw.text((right, panel_y + 170), f"Frame {last:02}", font=tiny, fill=_MUTED, anchor="rt")

    footer = height - 40
    draw.text((40, footer), "Prediction error", font=small, fill=_SCORE)
    draw.text((220, footer), f"Normal threshold: {threshold:.2e}", font=small, fill=_FAULT)
    draw.text((471, footer), "Shaded region: simulated event", font=small, fill=_MUTED)
    key = metrics["methods"]["latent_sigreg"]
    if "clip_ap" in key:
        draw.text((1160, footer), f"Held-out clip AP: {key['clip_ap']:.3f}", font=small, fill=_TEXT, anchor="rt")
    return image


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src 'sha256-@HASH@'; style-src 'unsafe-inline'">
<title>BeltWatch Lab | Replay</title>
<style>
:root{color-scheme:dark;--bg:#0a111d;--panel:#131f30;--line:#31445c;--text:#edf5ff;
      --muted:#a1b4cc;--mint:#73ecd8;--amber:#ffbb78;font:16px/1.5 system-ui,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text)}
main{max-width:1152px;margin:auto;padding:42px 24px 48px}
.eyebrow{margin:0;color:var(--mint);font:700 .77rem ui-monospace,monospace;letter-spacing:.15em}
h1{margin:10px 0 6px;font-size:clamp(2.1rem,6vw,3.8rem);line-height:1.07;letter-spacing:-.035em}
h2,p,figure{margin:0}
.description{max-width:66ch;color:var(--muted);margin-top:14px}
.replay{margin-top:26px;padding:18px;background:var(--panel);border:1px solid var(--line);border-radius:16px}
.controls{display:flex;align-items:end;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.field{display:grid;gap:4px}
label,.meta{color:var(--muted);font:600 .78rem ui-monospace,monospace}
button,select{min-height:42px;padding:8px 12px;background:var(--bg);color:var(--text);
              border:1px solid var(--line);border-radius:8px;font:inherit}
button{min-width:82px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--mint);outline-offset:3px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);gap:16px}
.pane{min-width:0;padding:13px;background:var(--bg);border:1px solid var(--line);border-radius:10px}
.pane h2{font-size:.95rem;margin-bottom:9px}
canvas{display:block;width:100%}
#camera{image-rendering:pixelated;aspect-ratio:3/2;border-radius:4px}
#chart{aspect-ratio:480/230}
figcaption,.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:8px;font-size:.79rem;color:var(--muted)}
.mint{color:var(--mint)}.amber{color:var(--amber)}
.timeline{margin-top:16px}
.timeline input{display:block;width:100%;margin:8px 0;accent-color:var(--mint)}
.metrics{display:flex;flex-wrap:wrap;gap:6px 28px;color:var(--muted);margin:18px 0 0;font-size:.85rem}
.metrics div{display:flex;gap:6px}.metrics dd{margin:0;color:var(--text);font-variant-numeric:tabular-nums}
footer{margin-top:18px;color:var(--muted);font-size:.84rem}
@media(max-width:720px){.grid{grid-template-columns:1fr}main{padding:28px 14px}.replay{padding:12px}}
</style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">BELTWATCH LAB &nbsp; / &nbsp; HELD-OUT REPLAY</p>
    <h1>When a conveyor surprises the model</h1>
    <p class="description">Replay synthetic clips and compare the model's forecast with its encoding of each observed
      frame. The threshold comes from a separate set of normal clips.</p>
  </header>
  <section class="replay" aria-label="Conveyor replay">
    <div class="controls">
      <div class="field"><label for="clip">Test clip</label><select id="clip"></select></div>
      <button id="play" type="button">Play</button>
      <span id="scene" class="meta"></span>
    </div>
    <div class="grid">
      <figure class="pane">
        <h2>Observed frame</h2>
        <canvas id="camera" role="img" aria-label="Synthetic grayscale conveyor frame"></canvas>
        <figcaption><span id="event"></span><span id="decision"></span></figcaption>
      </figure>
      <figure class="pane">
        <h2>Prediction error over time</h2>
        <canvas id="chart" width="960" height="460" role="img" aria-label="Prediction error by frame index"></canvas>
        <figcaption>
          <span class="mint">Solid: error</span><span class="amber">Dashed: normal threshold</span>
          <span>Shaded: simulated event</span>
        </figcaption>
      </figure>
    </div>
    <div class="timeline">
      <label for="frame">Frame <span id="position"></span></label>
      <input id="frame" type="range" min="0" value="0" step="1">
      <output for="frame" id="score" class="meta" aria-live="off"></output>
    </div>
    <dl id="metrics" class="metrics"></dl>
  </section>
  <footer>All footage is synthetic; this experiment has not been validated on real equipment.</footer>
  <noscript>Enable JavaScript to play the offline replay.</noscript>
</main>
<script type="application/json" id="run-data">@DATA@</script>
<script>@SCRIPT@</script>
</body>
</html>
"""

_SCRIPT = """
"use strict";
const $ = id => document.getElementById(id);
const data = JSON.parse($("run-data").textContent);
const clips = data.clips;
const method = data.methods.latent_sigreg;
const threshold = method.threshold;
const highest = clips.reduce((top, clip) =>
  clip.scores.reduce((peak, score) => score === null ? peak : Math.max(peak, score), top), threshold) || 1;
const fmt = n => n.toExponential(2);
const chooser = $("clip"), range = $("frame"), play = $("play");
const camera = $("camera").getContext("2d"), plot = $("chart").getContext("2d");
let selected, pixels, frame, timer = null;

clips.forEach((clip, i) => chooser.add(new Option(clip.fault.replace(/^./, c => c.toUpperCase()), i)));
for (const [key, label] of [
  ["clip_ap", "Held-out clip AP"], ["clip_auroc", "Clip AUROC"], ["frame_ap", "Frame AP"],
  ["recall", "Event recall"], ["false_alarm_rate", "Normal clips alerted"]
]) {
  if (typeof method[key] === "number") {
    const entry = document.createElement("div"), term = document.createElement("dt"), value = document.createElement("dd");
    term.textContent = label;
    value.textContent = method[key].toFixed(3);
    entry.append(term, value);
    $("metrics").append(entry);
  }
}

function stop() {
  clearInterval(timer);
  timer = null;
  play.textContent = "Play";
}

function choose() {
  stop();
  selected = clips[Number(chooser.value)];
  pixels = atob(selected.pixels);
  camera.canvas.width = selected.width;
  camera.canvas.height = selected.height;
  $("scene").textContent = "Scene " + String(selected.scene).padStart(4, "0");
  range.max = selected.scores.length - 1;
  range.value = 0;
  render();
}

function drawPlot(index) {
  const {width, height} = plot.canvas;
  const left = 100, right = width - 30, top = 34, bottom = height - 52;
  const x = i => left + i / (selected.scores.length - 1) * (right - left);
  const y = score => bottom - score / highest * (bottom - top);
  const line = (a, b, c, d) => {
    plot.beginPath(); plot.moveTo(a, b); plot.lineTo(c, d); plot.stroke();
  };
  plot.clearRect(0, 0, width, height);
  plot.fillStyle = "#302a2c";
  selected.labels.forEach((active, i) => {
    if (active) plot.fillRect(x(i), top, x(Math.min(i + 1, selected.labels.length - 1)) - x(i), bottom - top);
  });
  plot.strokeStyle = "#31445c";
  plot.lineWidth = 2;
  plot.font = "23px ui-monospace, monospace";
  plot.fillStyle = "#a1b4cc";
  [0, 0.5, 1].forEach(fraction => {
    const mark = highest * fraction;
    line(left, y(mark), right, y(mark));
    plot.fillText(mark === 0 ? "0" : fmt(mark), 0, y(mark) + 8);
  });
  const last = selected.scores.length - 1;
  plot.fillText("0", left, bottom + 34);
  plot.textAlign = "right";
  plot.fillText(String(last), right, bottom + 34);
  plot.textAlign = "left";
  plot.strokeStyle = "#ffbb78";
  plot.setLineDash([13, 9]);
  line(left, y(threshold), right, y(threshold));
  plot.setLineDash([]);
  plot.strokeStyle = "#73ecd8";
  plot.lineWidth = 4;
  plot.beginPath();
  selected.scores.forEach((score, i) => {
    if (score === null) return;
    if (i === data.context) plot.moveTo(x(i), y(score));
    else plot.lineTo(x(i), y(score));
  });
  plot.stroke();
  plot.strokeStyle = "#a1b4cc";
  plot.lineWidth = 2;
  line(x(index), top, x(index), bottom);
  const score = selected.scores[index];
  if (score !== null) {
    plot.fillStyle = "#73ecd8";
    plot.beginPath(); plot.arc(x(index), y(score), 7, 0, 2 * Math.PI); plot.fill();
  }
}

function render() {
  frame = Number(range.value);
  const image = camera.createImageData(selected.width, selected.height);
  const count = selected.width * selected.height;
  for (let i = 0; i < count; i++) {
    const gray = pixels.charCodeAt(frame * count + i);
    const offset = 4 * i;
    image.data.set([gray, gray, gray, 255], offset);
  }
  camera.putImageData(image, 0, 0);
  drawPlot(frame);
  const score = selected.scores[frame], active = selected.labels[frame];
  $("position").textContent = frame + " / " + range.max;
  $("event").textContent = active ? "Simulated " + selected.fault : "No simulated event";
  $("event").className = active ? "amber" : "";
  $("decision").textContent = score === null ? "Context" : score > threshold ? "Alert" : "No alert";
  $("decision").className = score !== null && score > threshold ? "amber" : "";
  const detail = score === null ? "Gathering " + data.context + " context frames"
    : "Error " + fmt(score) + " / threshold " + fmt(threshold);
  $("score").textContent = detail;
  range.setAttribute("aria-valuetext", "Frame " + frame + ". " + detail + ". " + $("event").textContent);
}

chooser.addEventListener("change", choose);
range.addEventListener("input", () => { stop(); render(); });
play.addEventListener("click", () => {
  if (timer !== null) { stop(); return; }
  if (frame === Number(range.max)) { range.value = 0; render(); }
  play.textContent = "Pause";
  timer = setInterval(() => {
    range.value = frame + 1;
    render();
    if (frame === Number(range.max)) stop();
  }, 125);
});
choose();
"""
