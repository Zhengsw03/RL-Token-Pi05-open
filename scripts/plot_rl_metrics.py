#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL-Token training-log curve visualizer.

Parses training logs into one curve per metric and renders a dependency-free,
offline-openable interactive HTML page (hover for values, moving-average
smoothing, log scale, CSV export).

Supported log formats (auto-detected):
  1. Stage 2 rl_metrics.log (TSV with a '#'-prefixed header)
       # updates  episode  buffer  q1_mean  q1_max  q1_rew  q1_norew  q2_mean ...
       Extra actor_on column (0/1): whether the RL Actor took part in the
       execution of the latest episode. The page offers an "Actor" selector:
       show all / highlight intervention bands (light-blue vertical bands) /
       show intervention segments only.
       Non-finite values are written as empty strings (never nan), so curves
       break wherever data is missing.
  2. Per-episode summary lines in Stage 2 train.log
       [RL] Q1=0.123 Q2=0.234 | Q1max=0.900 Q1rew=0.950 | TDtarget=0.910 | ...
       Missing fields are printed as '-' instead of nan; the parser treats them
       as gaps in the same way.
  3. Step lines in Stage 1 train.log
       Step 100/50000 | Loss: 1.2345 | LR: 1.00e-04 | 12.3 updates/s
  4. Stage 1 loss_history.json (list of floats)

Usage:
  # Render a static HTML file (offline-openable, shareable)
  python scripts/plot_rl_metrics.py <log-path> --out rl_metrics_plot.html

  # Live monitoring: local HTTP server, browser auto-refreshes the curves
  python scripts/plot_rl_metrics.py <log-path> --live --port 8000

  # Point at a stage-2 config and let it locate output_dir/rl_metrics.log
  python scripts/plot_rl_metrics.py --config configs/stage2_pi05.json --live
"""

import argparse
import json
import math
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

RE_STEP1 = re.compile(
    r"Step\s+(\d+)/(\d+)\s*\|\s*Loss:\s*([-+0-9.eE]+)\s*\|\s*"
    r"LR:\s*([-+0-9.eE]+)(?:\s*\|\s*([-+0-9.eE]+)\s*updates/s)?"
)
RE_RL_EP = re.compile(
    r"\[RL\]\s+Q1=([^\s|]+)\s+Q2=([^\s|]+)\s*\|\s*Q1max=([^\s|]+)\s+Q1rew=([^\s|]+)\s*\|\s*"
    r"TDtarget=([^\s|]+)\s*\|\s*TDerr=([^\s|]+)\s*\|\s*actorQ=([^\s|]+)\s*\|\s*"
    r"bc=([^\s|]+)\s*\|\s*nRew=([^\s|]+)\s*\|\s*updates=(\d+)\s*\|\s*buffer=(\d+)"
)


def _to_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def parse_tsv(path: Path):
    """Parse Stage2 rl_metrics.log. Returns (x column name, {column: [(x, y), ...]}, row count)."""
    header = None
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Only the FIRST '#' line is the TSV header; later '#' lines are
                # episode summary comments and must not overwrite it.
                if header is None:
                    header = [c.strip() for c in line.lstrip("#").split("\t") if c.strip()]
                continue
            if header is None:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rows.append(parts)
    if not header or not rows:
        return None
    x_name = header[0]
    series = {name: [] for name in header}
    for parts in rows:
        vals = [_to_float(v) for v in parts[: len(header)]]
        vals += [None] * (len(header) - len(vals))
        x = vals[0]
        if x is None:
            continue
        for name, v in zip(header, vals):
            series[name].append((x, v))
    return x_name, series, len(rows)


def parse_stage1(path: Path):
    """Parse Step/Loss/LR lines from a Stage1 train.log."""
    series = {"loss": [], "lr": [], "updates_per_sec": []}
    steps = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = RE_STEP1.search(raw)
            if not m:
                continue
            step, _total, loss, lr, ups = m.groups()
            x = int(step)
            steps += 1
            series["loss"].append((x, _to_float(loss)))
            series["lr"].append((x, _to_float(lr)))
            if ups:
                series["updates_per_sec"].append((x, _to_float(ups)))
    if steps == 0:
        return None
    # Merge loss_history.json from the same directory into a denser loss curve
    jp = path.parent / "loss_history.json"
    if jp.exists():
        try:
            vals = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(vals, list) and vals:
                series["loss_history"] = [(i, _to_float(v)) for i, v in enumerate(vals)]
        except Exception:
            pass
    return "step", series, steps


def parse_rl_episode(path: Path):
    """Parse [RL] per-episode summary lines from a Stage2 train.log."""
    cols = ["q1", "q2", "q1_max", "q1_rew", "td_target", "td_err",
            "actor_q", "bc", "n_rew", "buffer"]
    series = {c: [] for c in cols}
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = RE_RL_EP.search(raw)
            if not m:
                continue
            g = m.groups()
            x = int(g[9])  # updates is the x axis
            n += 1
            for i, c in enumerate(cols):
                if c == "buffer":
                    series[c].append((x, _to_float(g[10])))
                else:
                    series[c].append((x, _to_float(g[i])))
    if n == 0:
        return None
    return "updates", series, n


def parse_loss_json(path: Path):
    try:
        vals = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(vals, list) or not vals:
        return None
    cleaned = []
    for i, v in enumerate(vals):
        f = _to_float(v)
        if f is not None:
            cleaned.append((i, f))
    if not cleaned:
        return None
    return "step", {"loss": cleaned}, len(cleaned)


def parse_log(path: Path):
    """Auto-detect and parse a log file. Returns a dict, or None."""
    if not path.exists():
        return None
    # 1) loss_history.json
    if path.name == "loss_history.json" or path.suffix == ".json":
        r = parse_loss_json(path)
        if r:
            return {"mode": "loss_history.json", "x_name": r[0], "series": r[1],
                    "rows": r[2], "path": str(path)}
    # 2) Stage2 TSV metrics log
    r = parse_tsv(path)
    if r:
        x_name, series, rows = r
        return {"mode": "rl_metrics.log (TSV)", "x_name": x_name, "series": series,
                "rows": rows, "path": str(path)}
    # 3) Stage1 train.log
    r = parse_stage1(path)
    if r:
        x_name, series, rows = r
        return {"mode": "train.log (Stage 1 step lines)", "x_name": x_name, "series": series,
                "rows": rows, "path": str(path)}
    # 4) Stage2 train.log [RL] lines
    r = parse_rl_episode(path)
    if r:
        x_name, series, rows = r
        return {"mode": "train.log (Stage 2 [RL] per-episode)", "x_name": x_name, "series": series,
                "rows": rows, "path": str(path)}
    return None


def auto_locate_log(config_path=None):
    """Locate a log file by priority.

    When --config is given explicitly, output_dir/rl_metrics.log is the
    monitoring target: the path is returned even if the file does not exist yet
    (training not started; --live waits for it to appear). It never falls back
    to a discovered older log, which would monitor an unrelated experiment.
    """
    if config_path:
        cfg = Path(config_path)
        if cfg.exists():
            try:
                text = re.sub(r"//.*", "", cfg.read_text(encoding="utf-8"))
                data = json.loads(text)
                od = data.get("output_dir")
                if od:
                    return Path(od) / "rl_metrics.log"
            except Exception:
                pass
    candidates = []
    for base in (Path.cwd(), Path.cwd() / "outputs"):
        candidates.append(base / "rl_metrics.log")
        candidates.append(base / "train.log")
    for c in candidates:
        if c.exists():
            return c
    # Search the whole tree for the newest rl_metrics.log / train.log (hidden dirs included)
    newest, newest_mtime = None, -1.0
    for root, dirs, files in os.walk(Path.cwd()):
        for fname in files:
            if fname in ("rl_metrics.log", "train.log"):
                p = Path(root) / fname
                try:
                    mt = p.stat().st_mtime
                except OSError:
                    continue
                if mt > newest_mtime:
                    newest, newest_mtime = p, mt
    return newest


# ─────────────────────────────────────────────────────────────────────────────
# HTML template (zero external dependencies; embedded data + native Canvas drawing)
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RL-Token training curves - __MODE__</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
         background: #0b1220; color: #dbe4f0; padding: 18px; }
  h1 { font-size: 1.25em; margin-bottom: 4px; color: #7dd3fc; }
  .sub { color: #64748b; font-size: 0.8em; margin-bottom: 14px; word-break: break-all; }
  .bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 14px;
         background: #111c30; border: 1px solid #1e2a44; border-radius: 10px; padding: 10px 14px; }
  .bar label { font-size: 0.78em; color: #94a3b8; display: flex; align-items: center; gap: 6px; }
  select, button { background: #1e2a44; color: #dbe4f0; border: 1px solid #334155; border-radius: 6px;
                   padding: 5px 10px; font-size: 0.8em; cursor: pointer; }
  select:hover, button:hover { border-color: #38bdf8; }
  .stat { font-size: 0.8em; color: #7dd3fc; }
  .stat b { color: #e2e8f0; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }
  .card { background: #111c30; border: 1px solid #1e2a44; border-radius: 12px; padding: 10px 12px 6px; }
  .card-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
  .m-name { font-size: 0.85em; font-weight: 600; color: #e2e8f0; }
  .m-stat { font-size: 0.72em; color: #64748b; }
  canvas { width: 100%; height: 190px; display: block; cursor: crosshair; }
  .tip { position: fixed; pointer-events: none; background: #0b1220f2; border: 1px solid #38bdf8;
         color: #e2e8f0; font-size: 0.75em; padding: 6px 9px; border-radius: 8px; display: none;
         white-space: pre; z-index: 10; box-shadow: 0 4px 14px #000a; }
  .empty { color: #475569; font-size: 0.8em; padding: 30px; text-align: center; }
  .refresh-note { text-align: center; color: #475569; font-size: 0.72em; margin-top: 12px; }
  #toast { position: fixed; right: 16px; bottom: 16px; background: #166534; color: #bbf7d0;
           border-radius: 8px; padding: 8px 14px; font-size: 0.78em; display: none; }
</style>
</head>
<body>
<h1>RL-Token training curves - __MODE__</h1>
<div class="sub">Log: <span id="log-path">__PATH__</span> · rows: <span id="n-rows">0</span> ·
  x axis: <span id="x-name">-</span> · updated <span id="gen-time">-</span>
  <span id="live-badge" style="display:none"> · LIVE, auto-refresh every <span id="live-sec">3</span>s</span>
</div>

<div class="bar">
  <label>Smoothing <select id="smooth">
    <option value="1">off (raw)</option>
    <option value="5">5</option>
    <option value="20">20</option>
    <option value="50">50</option>
    <option value="100">100</option>
    <option value="200">200</option>
  </select></label>
  <label>Actor <select id="actorfilter">
    <option value="all">show all</option>
    <option value="highlight" selected>highlight intervention bands</option>
    <option value="active">intervention segments only</option>
  </select></label>
  <label><input type="checkbox" id="logscale"> log scale (Y)</label>
  <label><input type="checkbox" id="grid-toggle" checked> grid</label>
  <button id="csv-btn">Export CSV</button>
  <span class="stat">curves: <b id="n-series">0</b> · latest x: <b id="last-x">-</b></span>
</div>
<div class="sub" style="margin-top:0">Light-blue bands = RL Actor intervention periods (actor_on=1 ranges, i.e. episodes in which the Actor took over execution)</div>

<div class="grid" id="grid"></div>
<div id="toast">CSV exported</div>
<div class="refresh-note" id="refresh-note"></div>
<div class="tip" id="tip"></div>

<script id="chart-data" type="application/json">__DATA__</script>
<script>
"use strict";
window.__LIVE__ = false;
window.__LIVE_SEC__ = 3;
const PALETTE = ["#38bdf8", "#f472b6", "#4ade80", "#fbbf24", "#a78bfa", "#fb7185",
                 "#34d399", "#facc15", "#60a5fa", "#f97316", "#2dd4bf", "#e879f9"];
let DATA = null;

// -- Helpers --------------------------------------------------------
function smooth(vals, w) {
  if (w <= 1) return vals.slice();
  const out = new Array(vals.length);
  const half = Math.floor(w / 2);
  for (let i = 0; i < vals.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(vals.length - 1, i + half); j++) {
      const v = vals[j];
      if (v !== null && isFinite(v)) { s += v; n++; }
    }
    out[i] = n ? s / n : null;
  }
  return out;
}
function fmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  return v.toFixed(5);
}
function fmtX(v) {
  if (!isFinite(v)) return "—";
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return String(Math.round(v * 100) / 100);
}

// -- Single chart drawing -------------------------------------------
function drawChart(canvas, xs, ys, opt) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  if (W < 10) return;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = opt.grid ? "#1e2a44" : "transparent";
  ctx.lineWidth = 1;

  // all = every valid point (used for axis ranges); pts = filtered points to draw
  const all = [];
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (y !== null && isFinite(y) && xs[i] !== null && isFinite(xs[i])) all.push([xs[i], y]);
  }
  if (all.length < 2) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.textAlign = "center"; ctx.fillText("Not enough data points", W / 2, H / 2);
    return;
  }
  const pts = opt.actorMask ? all.filter(p => (opt.actorMask.get(p[0]) ?? 0) === 1) : all;
  if (pts.length < 2) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.textAlign = "center"; ctx.fillText("No data in Actor intervention segments", W / 2, H / 2);
    return;
  }
  const padL = 56, padR = 14, padT = 10, padB = 26;
  const pw = W - padL - padR, ph = H - padT - padB;

  let xMin = Infinity, xMax = -Infinity;
  for (const [x] of all) { xMin = Math.min(xMin, x); xMax = Math.max(xMax, x); }
  const xr = (xMax - xMin) || 1;
  const X = (x) => padL + ((x - xMin) / xr) * pw;

  // Intervention band highlight: light-blue vertical band + boundary lines
  if (opt.bands && opt.bands.length) {
    ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
    for (const [b0, b1] of opt.bands) {
      const x0 = Math.max(b0, xMin), x1 = Math.min(b1, xMax);
      if (x1 < x0) continue;
      ctx.fillRect(X(x0), padT, X(x1) - X(x0), ph);
    }
    ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
    ctx.lineWidth = 1;
    for (const [b0, b1] of opt.bands) {
      const x0 = Math.max(b0, xMin), x1 = Math.min(b1, xMax);
      if (x1 < x0) continue;
      ctx.beginPath(); ctx.moveTo(X(x0), padT); ctx.lineTo(X(x0), padT + ph); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(x1), padT); ctx.lineTo(X(x1), padT + ph); ctx.stroke();
    }
  }

  const logY = opt.logScale;
  let yMin = Infinity, yMax = -Infinity;
  const yv = [];
  for (const [, y] of pts) {
    if (logY) { if (y > 0) yv.push(y); }
    else yv.push(y);
  }
  if (logY && yv.length) { yMin = Math.min(...yv); yMax = Math.max(...yv); }
  else if (!logY && yv.length) {
    yMin = Math.min(...yv); yMax = Math.max(...yv);
    const span = (yMax - yMin) || Math.abs(yMax) * 0.1 || 1;
    yMin -= span * 0.08; yMax += span * 0.08;
  }
  if (!(yv.length && isFinite(yMin) && isFinite(yMax))) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.textAlign = "center"; ctx.fillText("No valid data", W / 2, H / 2);
    return;
  }
  const Y = (y) => {
    if (logY) return padT + ph - (Math.log10(y) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin) || 1) * ph;
    return padT + ph - (y - yMin) / ((yMax - yMin) || 1) * ph;
  };

  // Grid + Y ticks
  ctx.font = "10px sans-serif"; ctx.textAlign = "right";
  ctx.fillStyle = "#64748b";
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const t = i / ticks;
    const yv_tick = logY
      ? Math.pow(10, Math.log10(yMin) + t * (Math.log10(yMax) - Math.log10(yMin)))
      : yMin + t * (yMax - yMin);
    const yPix = padT + ph - t * ph;
    if (opt.grid) { ctx.beginPath(); ctx.moveTo(padL, yPix); ctx.lineTo(W - padR, yPix); ctx.stroke(); }
    ctx.fillText(fmt(yv_tick), padL - 6, yPix + 3);
  }
  // X ticks
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const xv = xMin + (xr / 4) * i;
    const xPix = X(xv);
    if (opt.grid) { ctx.beginPath(); ctx.moveTo(xPix, padT); ctx.lineTo(xPix, padT + ph); ctx.stroke(); }
    ctx.fillText(fmtX(xv), xPix, H - 8);
  }

  // Curve (breaks at NaN/None; in segments-only mode it breaks outside intervention ranges)
  const col = opt.color || "#38bdf8";
  ctx.strokeStyle = col; ctx.lineWidth = 1.6;
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.beginPath();
  let pen = false;
  for (const [x, y] of pts) {
    if (!logY || y > 0) {
      if (!pen) { ctx.moveTo(X(x), Y(y)); pen = true; }
      else ctx.lineTo(X(x), Y(y));
    } else pen = false;
  }
  ctx.stroke();

  // Highlight the last point
  const [lx, ly] = pts[pts.length - 1];
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(X(lx), Y(ly), 3, 0, Math.PI * 2); ctx.fill();
}

// -- Render all charts ----------------------------------------------
function render(force) {
  if (!DATA) return;
  const w = parseInt(document.getElementById("smooth").value, 10);
  const logScale = document.getElementById("logscale").checked;
  const grid = document.getElementById("grid-toggle").checked;
  const actorMode = document.getElementById("actorfilter").value;
  const gridEl = document.getElementById("grid");

  // actor_on gets no card of its own; it only defines the intervention bands
  const actSeries = DATA.series["actor_on"] || [];
  const names = Object.keys(DATA.series).filter(n => n !== "actor_on");
  document.getElementById("n-series").textContent = names.length;
  document.getElementById("n-rows").textContent = DATA.rows;
  document.getElementById("x-name").textContent = DATA.x_name;

  let lastX = null;
  for (const n of names) {
    const pts = DATA.series[n];
    if (pts.length && pts[pts.length - 1][0] !== null) lastX = pts[pts.length - 1][0];
  }
  document.getElementById("last-x").textContent = lastX === null ? "-" : fmtX(lastX);

  // Intervention bands: consecutive x ranges with actor_on==1 -> [[x0, x1], ...]
  const bands = [];
  {
    let start = null, prev = null;
    for (const [x, v] of actSeries) {
      if (v === 1) { if (start === null) start = x; prev = x; }
      else if (start !== null) { bands.push([start, prev]); start = null; }
    }
    if (start !== null) bands.push([start, prev]);
  }
  const actMap = new Map(actSeries);

  // Rebuild cards (only when the metric set changes)
  if (force || !gridEl._built || gridEl._builtNames !== names.join(",")) {
    gridEl.innerHTML = "";
    names.forEach((name, idx) => {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = '<div class="card-head"><span class="m-name"></span><span class="m-stat"></span></div>' +
                       '<canvas></canvas>';
      card.querySelector(".m-name").textContent = name;
      card.querySelector(".m-stat").textContent = "—";
      gridEl.appendChild(card);
    });
    gridEl._builtNames = names.join(",");
    gridEl._built = true;
  }

  names.forEach((name, idx) => {
    const card = gridEl.children[idx];
    const canvas = card.querySelector("canvas");
    const statEl = card.querySelector(".m-stat");
    const pts = DATA.series[name];
    const xs = pts.map(p => p[0]);
    const rawYs = pts.map(p => p[1]);
    const ys = smooth(rawYs, w);
    drawChart(canvas, xs, ys, {
      logScale, grid,
      color: PALETTE[idx % PALETTE.length],
      bands: actorMode === "all" ? [] : bands,
      actorMask: actorMode === "active" ? actMap : null,
    });
    // Stats: last valid value + mean
    let last = null, sum = 0, cnt = 0;
    for (const v of rawYs) { if (v !== null && isFinite(v)) { last = v; sum += v; cnt++; } }
    statEl.textContent = cnt ? ("last " + fmt(last) + " · avg " + fmt(sum / cnt)) : "no valid data";
    bindTip(canvas, xs, ys, name);
  });
}

// -- Hover tooltip --------------------------------------------------
function bindTip(canvas, xs, ys, name) {
  const tip = document.getElementById("tip");
  const onMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const xPix = e.clientX - rect.left;
    // Map the linear X pixel back to an index
    let best = -1, bestD = Infinity;
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] === null) continue;
      const frac = (xs[i] - (DATA.xMin ?? 0)) / ((DATA.xSpan ?? 1) || 1);
      const d = Math.abs(frac * rect.width - xPix);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best < 0 || bestD > rect.width / 30) { tip.style.display = "none"; return; }
    tip.style.display = "block";
    tip.style.left = (e.clientX + 12) + "px";
    tip.style.top = (e.clientY - 8) + "px";
    tip.textContent = name + "\n" + DATA.x_name + " = " + fmtX(xs[best]) + "\nvalue = " + fmt(ys[best]);
  };
  canvas.onmousemove = onMove;
  canvas.onmouseleave = () => { tip.style.display = "none"; };
}

// -- CSV export -----------------------------------------------------
function exportCSV() {
  if (!DATA) return;
  const names = Object.keys(DATA.series);
  const len = Math.max(...names.map(n => DATA.series[n].length));
  const rows = [["index", DATA.x_name, ...names]];
  const xmap = {};   // per column: x -> y (aligned by row)
  for (let c = 0; c < names.length; c++) {
    xmap[c] = new Map();
    for (const [x, y] of DATA.series[names[c]]) xmap[c].set(x, y);
  }
  const allX = [...new Set(names.flatMap(n => DATA.series[n].map(p => p[0]).filter(v => v !== null)))].sort((a, b) => a - b);
  for (const x of allX) {
    const row = [allX.indexOf(x), x];
    for (let c = 0; c < names.length; c++) row.push(xmap[c].has(x) ? xmap[c].get(x) : "");
    rows.push(row.map(v => (v === null || v === undefined) ? "" : v).join(","));
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "rl_metrics.csv";
  a.click();
  const toast = document.getElementById("toast");
  toast.style.display = "block";
  setTimeout(() => { toast.style.display = "none"; }, 1500);
}

// -- Initialization -------------------------------------------------
function loadData(cb) {
  if (window.__LIVE__) {
    fetch("/api/data").then(r => r.json()).then(d => { DATA = d; cb(); })
      .catch(() => setTimeout(() => loadData(cb), 2000));
  } else {
    const el = document.getElementById("chart-data");
    DATA = JSON.parse(el.textContent);
    cb();
  }
}

document.getElementById("smooth").addEventListener("change", () => render(false));
document.getElementById("actorfilter").addEventListener("change", () => render(false));
document.getElementById("logscale").addEventListener("change", () => render(false));
document.getElementById("grid").addEventListener("change", () => render(false));
document.getElementById("csv-btn").addEventListener("click", exportCSV);

loadData(() => {
  const hasSeries = !!(DATA && DATA.series && Object.keys(DATA.series).length);
  if (window.__LIVE__) {
    document.getElementById("live-badge").style.display = "inline";
    setInterval(() => loadData(() => render(false)), (window.__LIVE_SEC__ || 3) * 1000);
  }
  if (hasSeries) {
    // Precompute the X range for the tooltip
    let xMin = Infinity, xMax = -Infinity;
    for (const n of Object.keys(DATA.series)) {
      for (const [x] of DATA.series[n]) {
        if (x !== null && isFinite(x)) { xMin = Math.min(xMin, x); xMax = Math.max(xMax, x); }
      }
    }
    DATA.xMin = isFinite(xMin) ? xMin : 0;
    DATA.xSpan = isFinite(xMax - xMin) ? (xMax - xMin) : 1;
    document.getElementById("gen-time").textContent = DATA.gen_time || "-";
    render(true);
  } else if (window.__LIVE__) {
    // The log file (or its data rows) is still missing: wait for training to start;
    document.getElementById("gen-time").textContent = DATA && DATA.gen_time || "-";
    document.getElementById("grid").innerHTML =
      '<div class="empty">Waiting for training data...<br><br>Monitoring target: <code>' +
      (DATA && DATA.path ? DATA.path : "-") + '</code><br>' +
      'Curves appear automatically once training starts and the log shows up (no page reload needed for the initial render).</div>';
  } else {
    document.getElementById("grid").innerHTML = '<div class="empty">No metric curves were parsed. ' +
      'Check that the log format is supported (rl_metrics.log TSV / [RL] lines in train.log / Stage 1 Step lines).</div>';
  }
});
</script>
</body>
</html>
"""


def build_payload(parsed):
    """Turn parsed results into JSON-serializable data (NaN -> null, x column removed)."""
    series = {}
    for name, pts in parsed["series"].items():
        if name == parsed["x_name"]:
            continue
        if not pts:
            continue
        has_data = any(p[1] is not None for p in pts)
        if not has_data:
            continue
        series[name] = [[x, y] for x, y in pts]  # None is serialized as JSON null
    return {
        "mode": parsed["mode"],
        "path": parsed["path"],
        "x_name": parsed["x_name"],
        "rows": parsed["rows"],
        "gen_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "series": series,
    }


def render_html(payload):
    html = (HTML_TEMPLATE
            .replace("__MODE__", payload["mode"])
            .replace("__PATH__", payload["path"].replace("<", "&lt;").replace(">", "&gt;"))
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False)))
    return html


# ─────────────────────────────────────────────────────────────────────────────
# LIVE mode: HTTP server that the browser polls for /api/data
# ─────────────────────────────────────────────────────────────────────────────

class LiveHandler(BaseHTTPRequestHandler):
    payload = None
    log_path = None
    interval = 3

    def do_GET(self):
        if self.path == "/api/data":
            parsed = parse_log(Path(self.log_path))
            self.payload = build_payload(parsed) if parsed else {
                "mode": "log not found", "path": str(self.log_path), "x_name": "-",
                "rows": 0, "gen_time": time.strftime("%Y-%m-%d %H:%M:%S"), "series": {},
            }
            body = json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            html = (HTML_TEMPLATE
                    .replace("__MODE__", "LIVE monitoring")
                    .replace("__PATH__", str(self.log_path).replace("<", "&lt;").replace(">", "&gt;"))
                    .replace("__DATA__", "{}"))
            html = html.replace("window.__LIVE__ = false;",
                                "window.__LIVE__ = true; window.__LIVE_SEC__ = %g;" % self.interval)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

    def log_message(self, fmt, *args):
        # Silence the /api/data polling access log (one line every few seconds is noise);
        # errors are still printed
        if self.path.startswith("/api/data"):
            return
        sys.stdout.write("[live] %s\n" % (fmt % args))


def main():
    ap = argparse.ArgumentParser(description="RL-Token training-log curve visualizer (dependency-free HTML)")
    ap.add_argument("log", nargs="?", default=None,
                    help="log file path; when omitted, rl_metrics.log / train.log are auto-located")
    ap.add_argument("--config", default=None,
                    help="Stage 2 config file (JSONC); reads output_dir/rl_metrics.log from it")
    ap.add_argument("--out", default=None,
                    help="output HTML path (default <log-dir>/rl_metrics_plot.html)")
    ap.add_argument("--live", action="store_true", help="start the live HTTP monitoring server")
    ap.add_argument("--port", type=int, default=8000, help="LIVE-mode port (default 8000)")
    ap.add_argument("--interval", type=float, default=3.0, help="LIVE-mode refresh interval in seconds (default 3)")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else auto_locate_log(args.config)
    if args.live:
        # LIVE mode tolerates a missing log (training not started yet): the page
        # shows a waiting message and curves appear once the file exists. The
        # target given by --config or an explicit path is the monitoring target;
        # failing to find any path at all is the only error case.
        if log_path is None:
            sys.exit("error: --live found no log path. Pass `python scripts/plot_rl_metrics.py "
                     "<log-path> --live` or `--config <stage2-config> --live`.")
        if not log_path.exists():
            print(f"note: {log_path} does not exist yet (training may not have started); the page will wait for it...")
        out = Path(args.out) if args.out else log_path.parent / "rl_metrics_plot.html"
        handler = LiveHandler
        handler.log_path = str(log_path)
        handler.interval = args.interval
        server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
        print(f"LIVE monitoring: http://127.0.0.1:{args.port}")
        print(f"monitoring log: {log_path}")
        print("Press Ctrl+C to stop")
        # Also write a static snapshot
        parsed = parse_log(log_path) if log_path.exists() else None
        if parsed:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_html(build_payload(parsed)), encoding="utf-8")
            print(f"static snapshot: {out}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    if log_path is None or not log_path.exists():
        sys.exit("error: no log file found. Pass `python scripts/plot_rl_metrics.py <log-path>` "
                 "or use --config <stage2-config> to locate it.")

    parsed = parse_log(log_path)
    if parsed is None:
        sys.exit(f"error: unrecognized log format: {log_path}\n"
                 f"supported: rl_metrics.log (TSV) / [RL] lines in train.log / Stage 1 Step lines / loss_history.json")
    out = Path(args.out) if args.out else log_path.parent / "rl_metrics_plot.html"
    out.write_text(render_html(build_payload(parsed)), encoding="utf-8")
    print(f"parsed {len(parsed['series'])} metric(s) and {parsed['rows']} data row(s): {parsed['mode']}")
    print(f"written: {out}  (open it directly in a browser)")
    print("note: add --live to start live monitoring.")


if __name__ == "__main__":
    main()
