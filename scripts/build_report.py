"""
build_report.py — generate a static multi-page HTML report from a sweep run.

Reads every trajectory JSON under runs/{run_name}/{model}/ and emits:

    runs/{run_name}/report/
        index.html               -- overall summary + per-model breakdown
        models/{model}.html      -- grid of trajectories for one model
        t/{model}__l{L}_p{P}.html -- one page per trajectory (TLDR + conversation)
        style.css                -- shared styles (bv-registry-inspired)

Safe to call repeatedly while a sweep is in progress — each call rebuilds the
whole site from disk.

Usage:
    uv run python scripts/build_report.py runs/my_sweep
"""

from __future__ import annotations

import glob
import html
import json
import os
import sys
from typing import Any

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Style — distilled from mlab/bv-registry/web/style.css
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --g: #00553A;
  --w: #ffffff;
  --r: #aa2222;
  --y: #b08900;
  --g08: rgba(0,85,58,0.08);
  --g15: rgba(0,85,58,0.15);
  --g35: rgba(0,85,58,0.35);
  --t: 80ms ease;
  --max: 1100px;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}

html { background: var(--w); }
body {
  background: var(--w); color: var(--g);
  font-family: var(--mono);
  font-size: 14px; line-height: 1.55; min-height: 100vh;
}
a { color: var(--g); }
a:hover { opacity: 0.6; }

.page-wrap { max-width: var(--max); margin: 0 auto; padding: 0 24px; }

header { border-bottom: 1px solid var(--g); }
header .page-wrap { padding-top: 36px; padding-bottom: 24px; }
.site-title {
  font-weight: 400; font-style: italic; font-size: 36px;
  letter-spacing: -0.02em; line-height: 1; margin-bottom: 6px;
}
.site-subtitle { font-size: 13px; opacity: 0.45; font-style: italic; }
.header-links { margin-top: 12px; display: flex; gap: 14px; font-size: 12px; font-style: italic; }
.header-links a { opacity: 0.55; text-decoration: none; }
.header-links a:hover { opacity: 1; }
.header-links .meta { opacity: 0.32; }

/* sitemap nav — appears on every page so any (variant, model) is one click away */
.sitemap {
  border-bottom: 1px solid var(--g15);
  background: var(--g08);
  font-size: 11px;
}
.sitemap .page-wrap { padding-top: 10px; padding-bottom: 10px; }
.sitemap .group { margin-right: 18px; display: inline-block; }
.sitemap .grp-label {
  font-style: italic; opacity: 0.55;
  letter-spacing: 0.06em; margin-right: 6px;
}
.sitemap a {
  text-decoration: none; opacity: 0.75;
  margin-right: 8px;
}
.sitemap a:hover { opacity: 1; }
.sitemap a.current { opacity: 1; font-weight: 500; text-decoration: underline; }

/* summary stats */
.stat-row {
  display: flex; gap: 0; border-bottom: 1px solid var(--g);
}
.stat {
  flex: 1; padding: 18px 20px; border-right: 1px solid var(--g15);
}
.stat:last-child { border-right: none; }
.stat-label {
  font-size: 11px; font-style: italic; opacity: 0.45;
  letter-spacing: 0.07em; margin-bottom: 4px;
}
.stat-val { font-size: 22px; letter-spacing: -0.01em; }
.stat-val.ok { color: #2d6a4f; }
.stat-val.bad { color: var(--r); }

/* model rollup table */
.model-table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0;
}
.model-table th, .model-table td {
  padding: 9px 14px; text-align: left;
  border-bottom: 1px solid var(--g15); font-size: 13px;
}
.model-table th {
  font-weight: 500; font-size: 11px; font-style: italic;
  letter-spacing: 0.07em; opacity: 0.55;
  border-bottom: 1px solid var(--g);
}
.model-table tr:hover td { background: var(--g08); }
.model-table a { text-decoration: none; }

/* card grid */
.tool-grid {
  display: grid; grid-template-columns: repeat(4, 1fr);
  background: var(--g); gap: 1px; padding-bottom: 1px;
  margin-top: 18px;
}
.tool-card {
  background: var(--w); padding: 14px 16px; cursor: pointer;
  transition: background var(--t), color var(--t);
  display: flex; flex-direction: column; gap: 6px;
  text-decoration: none; color: var(--g);
}
.tool-card:hover { background: var(--g); color: var(--w); }
.card-top { display: flex; align-items: baseline; gap: 8px; }
.card-name { font-size: 13px; letter-spacing: -0.01em; }
.card-meta { font-size: 11px; opacity: 0.45; font-style: italic; margin-left: auto; }

.outcome-badge {
  font-size: 10px; padding: 1px 6px; border: 1px solid currentColor;
  letter-spacing: 0.04em; font-style: italic; opacity: 0.7;
}
.outcome-correct { color: #2d6a4f; }
.outcome-incorrect { color: var(--y); }
.outcome-compile_fail { color: var(--r); }
.outcome-error { color: var(--r); }
.outcome-skipped { opacity: 0.45; }
.outcome-unknown { opacity: 0.4; }
.outcome-in_progress { color: var(--g); font-style: italic; }
.outcome-in_progress::after { content: " ●"; animation: blink 1.4s infinite; }
@keyframes blink { 0%,50%,100% { opacity: 1; } 25%,75% { opacity: 0.25; } }

.card-row { display: flex; gap: 8px; align-items: baseline; }
.card-runtime { font-size: 11px; opacity: 0.55; font-style: italic; }
.card-speedup { font-size: 11px; }
.speedup-pos { color: #2d6a4f; }
.speedup-neg { color: var(--y); }

/* trajectory page */
.tldr {
  background: var(--g08); border: 1px solid var(--g15);
  padding: 16px 18px; margin: 18px 0;
}
.tldr-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  margin-top: 8px;
}
.tldr-cell .l { font-size: 10px; opacity: 0.45; font-style: italic; letter-spacing: 0.07em; }
.tldr-cell .v { font-size: 14px; }

/* extended metrics block */
.metrics {
  border: 1px solid var(--g15); margin: 18px 0;
}
.metrics-section {
  border-bottom: 1px solid var(--g15);
}
.metrics-section:last-child { border-bottom: none; }
.metrics-section > .metrics-head {
  background: var(--g08); padding: 8px 14px; font-size: 11px;
  font-style: italic; letter-spacing: 0.07em; opacity: 0.7;
  display: flex; justify-content: space-between; align-items: baseline;
}
.metrics-section > .metrics-head .src {
  font-size: 10px; opacity: 0.55;
}
.metrics-grid {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--g15);
}
.metric {
  background: var(--w); padding: 8px 12px;
  display: flex; flex-direction: column; gap: 2px;
}
.metric .l { font-size: 10px; opacity: 0.45; font-style: italic; letter-spacing: 0.07em; }
.metric .v { font-size: 13px; }
.metric .v.muted { opacity: 0.45; font-style: italic; }
.metric .v.ratio-good { color: #2d6a4f; }
.metric .v.ratio-bad { color: var(--y); }

/* sweep-wide rollup table */
.metrics-rollup {
  width: 100%; border-collapse: collapse; margin: 0 0 6px 0;
}
.metrics-rollup td {
  padding: 9px 14px; font-size: 12px;
  border-bottom: 1px solid var(--g15);
}
.metrics-rollup td:first-child {
  width: 38%; opacity: 0.55; font-style: italic;
}

.turn {
  border-top: 1px solid var(--g15); padding: 16px 0;
}
.turn-head {
  display: flex; align-items: baseline; gap: 12px;
  font-size: 12px; opacity: 0.7; margin-bottom: 8px;
}
.turn-head .turn-id { font-size: 13px; opacity: 1; }

details { margin: 6px 0; }
details > summary {
  cursor: pointer; font-size: 12px; opacity: 0.65;
  list-style: none; padding: 4px 0;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸ "; opacity: 0.5; }
details[open] > summary::before { content: "▾ "; }

.block {
  background: var(--g08); border-left: 2px solid var(--g35);
  padding: 8px 12px; margin: 4px 0;
  white-space: pre-wrap; word-break: break-word;
  font-size: 12px;
}
.block.tool-args { border-left-color: var(--g); }
.block.tool-out { border-left-color: #2d6a4f; }
.block.tool-out.fail { border-left-color: var(--r); }
.block.reasoning { border-left-color: var(--y); font-style: italic; opacity: 0.85; }
.block.assistant { border-left-color: var(--g); }

.role-tag {
  display: inline-block;
  font-size: 10px; padding: 0 5px; margin-right: 6px;
  border: 1px solid currentColor; letter-spacing: 0.04em;
  font-style: italic; opacity: 0.6; vertical-align: middle;
}

@media (max-width: 900px) {
  .tool-grid { grid-template-columns: repeat(2, 1fr); }
  .stat-row { flex-direction: column; }
  .stat { border-right: none; border-bottom: 1px solid var(--g15); }
}
"""


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def _page(title: str, body: str, css_path: str = "../style.css") -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{css_path}">
<meta http-equiv="refresh" content="60">
</head><body>
{body}
</body></html>"""


def _header(run_name: str, generated_at: str, here_link: str = "index.html",
            link_prefix: str = "") -> str:
    return f"""<header><div class="page-wrap">
<div class="site-title">{html.escape(run_name)}</div>
<div class="site-subtitle">kernelbench sweep report</div>
<div class="header-links">
  <a href="{link_prefix}{here_link}">overview</a>
  <span class="meta">generated {html.escape(generated_at)}</span>
</div>
</div></header>"""


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

def _load_all_trajectories(run_dir: str) -> list[dict]:
    """Load every trajectory JSON under the run directory.

    Supports two layouts:
      - new: runs/{name}/{variant}/{model}/level_*_problem_*_trajectory.json
      - legacy: runs/{name}/{model}/level_*_problem_*_trajectory.json
        (variant is set to "default" for legacy runs)
    """
    out = []
    seen = set()
    patterns = [
        (os.path.join(run_dir, "*", "*", "level_*_problem_*_trajectory.json"), 2),
        (os.path.join(run_dir, "*", "level_*_problem_*_trajectory.json"), 1),
    ]
    for pattern, depth in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            # Skip the report directory if its name happens to match the glob.
            rel_parts = os.path.relpath(path, run_dir).split(os.sep)
            if rel_parts and rel_parts[0] == "report":
                continue
            seen.add(path)
            try:
                with open(path) as f:
                    d = json.load(f)
            except Exception as e:
                print(f"[build_report] could not read {path}: {e}")
                continue
            d["_path"] = path
            d["_model_dir"] = os.path.basename(os.path.dirname(path))
            d["_variant"] = rel_parts[0] if depth == 2 else "default"
            out.append(d)
    return out


def _outcome_class(outcome: str) -> str:
    return f"outcome-{outcome}" if outcome else "outcome-unknown"


def _speedup(d: dict) -> float | None:
    fr = d.get("final_result") or {}
    rt = fr.get("runtime")
    rrt = fr.get("ref_runtime")
    if rt and rrt and rt > 0 and rrt > 0:
        return float(rrt) / float(rt)
    return None


def _final_metric(d: dict, group: str, key: str) -> float | None:
    """Pull a single numeric metric out of final_result.{group}[key]; None if missing/sentinel."""
    fr = d.get("final_result") or {}
    g = fr.get(group) or {}
    if not isinstance(g, dict):
        return None
    v = g.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v < 0 or v == float("inf"):
        return None
    return v


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate_metrics(trajs: list[dict]) -> dict:
    """Compute sweep-wide averages over the extended metrics, restricted to
    correct kernels (so memory/energy ratios aren't polluted by failed runs).
    """
    correct = [d for d in trajs if d.get("outcome") == "correct"]
    speedups = [s for s in (_speedup(d) for d in correct) if s is not None]
    mem_ratios = [v for v in (_final_metric(d, "memory_stats", "memory_ratio") for d in correct) if v is not None]
    fusion = [v for v in (_final_metric(d, "kernel_launch_stats", "fusion_ratio") for d in correct) if v is not None]
    energy = [v for v in (_final_metric(d, "energy_stats", "energy_ratio") for d in correct) if v is not None]
    sol = [v for v in (_final_metric(d, "sol_stats", "sol_score") for d in correct) if v is not None]
    max_abs = [v for v in (_final_metric(d, "numerical_precision", "max_abs_error") for d in correct) if v is not None]
    occ = [v for v in (_final_metric(d, "roofline_stats", "occupancy_pct") for d in correct) if v is not None]
    dram = [v for v in (_final_metric(d, "roofline_stats", "dram_utilization_pct") for d in correct) if v is not None]
    return {
        "n_correct": len(correct),
        "avg_speedup": _avg(speedups),
        "avg_memory_ratio": _avg(mem_ratios),
        "avg_fusion_ratio": _avg(fusion),
        "avg_energy_ratio": _avg(energy),
        "avg_sol": _avg(sol),
        "avg_max_abs_err": _avg(max_abs),
        "avg_occupancy_pct": _avg(occ),
        "avg_dram_util_pct": _avg(dram),
        "n_with_memory": len(mem_ratios),
        "n_with_fusion": len(fusion),
        "n_with_energy": len(energy),
        "n_with_sol": len(sol),
    }


def _render_sitemap(by_variant: dict[str, list[dict]], *, link_prefix: str,
                    here: tuple[str, str | None] = ("top", None)) -> str:
    """Render a single-row sitemap with a Home link and per-variant model lists.

    `link_prefix` is the relative path back to the report root, e.g. ""
    (top index), "../../" (variant index), "../" (model page is one level up
    from variant root via models/, but href targets need to be from the
    *page* — see callers).

    `here` is ('top'|'variant'|'model'|'trajectory', context) used to
    highlight the current page.
    """
    parts = [
        '<nav class="sitemap"><div class="page-wrap">',
        '<span class="group">'
        f'<a href="{link_prefix}index.html"'
        f'{" class=current" if here[0] == "top" else ""}>home</a></span>',
    ]
    for variant in sorted(by_variant.keys()):
        v_safe = _safe_filename(variant)
        v_href = f'{link_prefix}v/{v_safe}/index.html'
        v_cur = (here[0] == "variant" and here[1] == variant)
        models = sorted({d.get("model_name") or d["_model_dir"]
                         for d in by_variant[variant]})
        model_links = []
        for m in models:
            m_safe = _safe_filename(m)
            m_href = f'{link_prefix}v/{v_safe}/models/{m_safe}.html'
            cur = (here[0] in ("model", "trajectory")
                   and here[1] == (variant, m))
            model_links.append(
                f'<a href="{m_href}"{" class=current" if cur else ""}>'
                f'{html.escape(m)}</a>'
            )
        parts.append(
            f'<span class="group"><span class="grp-label">{html.escape(variant)}:</span>'
            f'<a href="{v_href}"{" class=current" if v_cur else ""}>overview</a>'
            f'{"".join(model_links)}</span>'
        )
    parts.append('</div></nav>')
    return "".join(parts)


def _outcome_counts(trajs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in trajs:
        o = d.get("outcome") or "unknown"
        out[o] = out.get(o, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Index / model pages
# ---------------------------------------------------------------------------

def _stat_row(trajs: list[dict]) -> str:
    total = len(trajs)
    n_correct = sum(1 for d in trajs if d.get("outcome") == "correct")
    n_compiled = sum(
        1 for d in trajs if (d.get("final_result") or {}).get("compiled")
    )
    n_done = sum(1 for d in trajs if d.get("finished_at"))
    agg = _aggregate_metrics(trajs)
    return f"""<div class="stat-row">
  <div class="stat"><div class="stat-label">trajectories</div>
    <div class="stat-val">{total}</div></div>
  <div class="stat"><div class="stat-label">finished</div>
    <div class="stat-val">{n_done}</div></div>
  <div class="stat"><div class="stat-label">correct</div>
    <div class="stat-val ok">{n_correct}</div></div>
  <div class="stat"><div class="stat-label">compiled</div>
    <div class="stat-val">{n_compiled}</div></div>
  <div class="stat"><div class="stat-label">avg speedup (correct)</div>
    <div class="stat-val">{agg['avg_speedup']:.2f}x</div></div>
</div>"""


def _sweep_metrics_block(trajs: list[dict]) -> str:
    """Sweep-wide rollup of extended eval metrics, averaged over correct items."""
    agg = _aggregate_metrics(trajs)
    counts = _outcome_counts(trajs)
    if agg["n_correct"] == 0 and not counts:
        return ""

    def _row(label: str, val: str, note: str = "") -> str:
        note_html = (f' <span style="opacity:0.45;font-style:italic;">({html.escape(note)})</span>'
                     if note else "")
        return f"<tr><td>{html.escape(label)}</td><td>{val}{note_html}</td></tr>"

    nc = agg["n_correct"]
    rows = [
        _row("outcomes",
             " · ".join(f'<span class="outcome-badge {_outcome_class(o)}">{o} {n}</span>'
                        for o, n in sorted(counts.items(), key=lambda x: -x[1]))),
        _row("avg speedup", f"{agg['avg_speedup']:.2f}x", f"over {nc} correct"),
        _row("avg memory ratio (kernel / ref)",
             f"{agg['avg_memory_ratio']:.2f}x" if agg["n_with_memory"] else "—",
             f"{agg['n_with_memory']} samples"),
        _row("avg fusion ratio (ref launches / kernel launches)",
             f"{agg['avg_fusion_ratio']:.2f}x" if agg["n_with_fusion"] else "—",
             f"{agg['n_with_fusion']} samples"),
        _row("avg energy ratio (ref / kernel)",
             f"{agg['avg_energy_ratio']:.2f}x" if agg["n_with_energy"] else "—",
             f"{agg['n_with_energy']} samples"),
        _row("avg SOL score (0–1)",
             f"{agg['avg_sol']:.3f}" if agg["n_with_sol"] else "—",
             f"{agg['n_with_sol']} samples"),
        _row("avg DRAM util %",
             f"{agg['avg_dram_util_pct']:.1f}%" if agg["avg_dram_util_pct"] else "—"),
        _row("avg occupancy %",
             f"{agg['avg_occupancy_pct']:.1f}%" if agg["avg_occupancy_pct"] else "—"),
        _row("avg max abs error",
             f"{agg['avg_max_abs_err']:.2e}" if agg["avg_max_abs_err"] else "—"),
    ]
    body = (
        '<div class="metrics-section">'
        '<div class="metrics-head"><span>sweep-wide eval metrics</span>'
        '<span class="src">averaged over correct kernels</span></div>'
        f'<table class="metrics-rollup"><tbody>{"".join(rows)}</tbody></table>'
        '</div>'
    )
    return f'<div class="page-wrap"><div class="metrics">{body}</div></div>'


def _render_top_index(run_name: str, by_variant: dict[str, list[dict]],
                      generated_at: str) -> str:
    """Top-level overview that lists each variant as a card-grid entry."""
    all_trajs = [d for v in by_variant.values() for d in v]

    cards = []
    for variant in sorted(by_variant.keys()):
        items = by_variant[variant]
        n = len(items)
        n_correct = sum(1 for d in items if d.get("outcome") == "correct")
        n_compiled = sum(1 for d in items if (d.get("final_result") or {}).get("compiled"))
        speedups = [s for s in (_speedup(d) for d in items) if s is not None]
        avg = sum(speedups) / len(speedups) if speedups else 0.0
        models = sorted({d.get("model_name") or d["_model_dir"] for d in items})
        cards.append(f"""<a class="tool-card" href="v/{_safe_filename(variant)}/index.html">
  <div class="card-top">
    <span class="card-name">{html.escape(variant)}</span>
    <span class="card-meta">{n} trajectories · {len(models)} models</span>
  </div>
  <div class="card-row">
    <span class="outcome-badge outcome-correct">{n_correct} correct</span>
    <span class="card-runtime">{n_compiled} compiled · avg {avg:.2f}x</span>
  </div>
</a>""")

    head = (
        f'<header><div class="page-wrap">'
        f'<div class="site-title">{html.escape(run_name)}</div>'
        f'<div class="site-subtitle">kernelbench sweep report · {len(by_variant)} variants</div>'
        f'<div class="header-links">'
        f'<a href="index.html">overview</a>'
        f'<span class="meta">generated {html.escape(generated_at)}</span>'
        f'</div></div></header>'
    )
    body = (
        head
        + _render_sitemap(by_variant, link_prefix="", here=("top", None))
        + _stat_row(all_trajs)
        + _sweep_metrics_block(all_trajs)
        + f'<div class="page-wrap"><div class="tool-grid">{"".join(cards)}</div></div>'
    )
    return body


def _render_variant_index(run_name: str, variant: str, trajs: list[dict],
                          generated_at: str,
                          by_variant: dict[str, list[dict]] | None = None) -> str:
    """Per-variant summary page (used to be the top-level index)."""
    by_model: dict[str, list[dict]] = {}
    for d in trajs:
        by_model.setdefault(d.get("model_name") or d["_model_dir"], []).append(d)

    rows = []
    for model in sorted(by_model.keys()):
        items = by_model[model]
        m_total = len(items)
        m_correct = sum(1 for d in items if d.get("outcome") == "correct")
        m_compiled = sum(1 for d in items if (d.get("final_result") or {}).get("compiled"))
        m_speedups = [s for s in (_speedup(d) for d in items) if s is not None]
        m_avg = sum(m_speedups) / len(m_speedups) if m_speedups else 0.0
        rows.append(f"""<tr>
  <td><a href="models/{html.escape(_safe_filename(model))}.html">{html.escape(model)}</a></td>
  <td>{m_total}</td>
  <td>{m_correct}</td>
  <td>{m_compiled}</td>
  <td>{m_avg:.2f}x</td>
</tr>""")

    table = f"""<table class="model-table">
<thead><tr><th>model</th><th>n</th><th>correct</th><th>compiled</th>
<th>avg speedup</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>"""

    head = (
        f'<header><div class="page-wrap">'
        f'<div class="site-title">{html.escape(variant)}</div>'
        f'<div class="site-subtitle">{html.escape(run_name)} · variant overview</div>'
        f'<div class="header-links">'
        f'<a href="../../index.html">← all variants</a>'
        f'<span class="meta">generated {html.escape(generated_at)}</span>'
        f'</div></div></header>'
    )
    sitemap = (_render_sitemap(by_variant, link_prefix="../../",
                               here=("variant", variant))
               if by_variant else "")
    body = (
        head
        + sitemap
        + _stat_row(trajs)
        + _sweep_metrics_block(trajs)
        + f'<div class="page-wrap">{table}</div>'
    )
    return body


def _render_model_page(model: str, variant: str, trajs: list[dict],
                       run_name: str, generated_at: str,
                       by_variant: dict[str, list[dict]] | None = None) -> str:
    cards = []
    for d in sorted(
        trajs, key=lambda x: (x.get("level", 0), x.get("problem_id", 0))
    ):
        level = d.get("level")
        pid = d.get("problem_id")
        outcome = d.get("outcome", "unknown")
        sp = _speedup(d)
        sp_str = ""
        if sp is not None:
            cls = "speedup-pos" if sp >= 1 else "speedup-neg"
            sp_str = f'<span class="card-speedup {cls}">{sp:.2f}x</span>'
        rt = (d.get("final_result") or {}).get("runtime", -1)
        rt_str = f"{rt:.1f}μs" if rt and rt > 0 else "—"
        traj_id = _trajectory_id(d)
        cards.append(f"""<a class="tool-card" href="../t/{traj_id}.html">
  <div class="card-top">
    <span class="card-name">L{level} P{pid}</span>
    <span class="card-meta">{html.escape(d.get('problem_name','')[:32])}</span>
  </div>
  <div class="card-row">
    <span class="outcome-badge {_outcome_class(outcome)}">{outcome}</span>
    {sp_str}
    <span class="card-runtime">{rt_str}</span>
  </div>
</a>""")

    body = (
        f'<header><div class="page-wrap">'
        f'<div class="site-title">{html.escape(model)}</div>'
        f'<div class="site-subtitle">{html.escape(run_name)} · {html.escape(variant)} · '
        f'{len(trajs)} trajectories</div>'
        f'<div class="header-links">'
        f'<a href="../../../index.html">← all variants</a>'
        f'<a href="../index.html">← {html.escape(variant)}</a>'
        f'<span class="meta">generated {html.escape(generated_at)}</span>'
        f'</div></div></header>'
        + (_render_sitemap(by_variant, link_prefix="../../../",
                           here=("model", (variant, model))) if by_variant else "")
        + f'<div class="page-wrap"><div class="tool-grid">{"".join(cards)}</div></div>'
    )
    return body


# ---------------------------------------------------------------------------
# Trajectory page
# ---------------------------------------------------------------------------

def _trajectory_id(d: dict) -> str:
    model = d.get("model_name") or d["_model_dir"]
    return _safe_filename(f"{model}__l{d.get('level')}_p{d.get('problem_id')}")


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _render_trajectory(d: dict, run_name: str, generated_at: str,
                       by_variant: dict[str, list[dict]] | None = None) -> str:
    fr = d.get("final_result") or {}
    sp = _speedup(d)
    sp_str = f"{sp:.2f}x" if sp is not None else "—"
    rt = fr.get("runtime", -1)
    rt_str = f"{rt:.2f} μs" if rt and rt > 0 else "—"

    tldr = f"""<div class="tldr">
<div class="card-row">
  <span class="outcome-badge {_outcome_class(d.get('outcome',''))}">{d.get('outcome','?')}</span>
  <span style="font-size:13px;">{html.escape(d.get('problem_name',''))}</span>
</div>
<div class="tldr-grid">
  <div class="tldr-cell"><div class="l">model</div><div class="v">{html.escape(d.get('model_name',''))}</div></div>
  <div class="tldr-cell"><div class="l">turns / max</div><div class="v">{d.get('total_turns',0)} / {d.get('max_turns','?')}</div></div>
  <div class="tldr-cell"><div class="l">tool calls</div><div class="v">{d.get('total_tool_calls',0)}</div></div>
  <div class="tldr-cell"><div class="l">backend</div><div class="v">{html.escape(d.get('backend',''))} {html.escape(d.get('precision',''))}</div></div>
  <div class="tldr-cell"><div class="l">variant</div><div class="v">{html.escape(d.get('_variant','default'))}</div></div>
  <div class="tldr-cell"><div class="l">runtime</div><div class="v">{rt_str}</div></div>
  <div class="tldr-cell"><div class="l">ref runtime</div>
    <div class="v">{(fr.get('ref_runtime') or 0):.2f} μs</div></div>
  <div class="tldr-cell"><div class="l">speedup</div><div class="v">{sp_str}</div></div>
  <div class="tldr-cell"><div class="l">started</div><div class="v">{html.escape(d.get('started_at','—'))}</div></div>
</div>
{('<div style="margin-top:10px;font-size:12px;opacity:.7;"><b>compile error:</b> ' + html.escape(str(fr.get('metadata',{}).get('compilation_error',''))[:600]) + '</div>') if not fr.get('compiled', True) else ''}
{('<div style="margin-top:10px;font-size:12px;opacity:.7;"><b>correctness issue:</b> ' + html.escape(str(fr.get('metadata',{}).get('correctness_issue',''))[:600]) + '</div>') if fr.get('compiled') and not fr.get('correctness') else ''}
{('<div style="margin-top:10px;font-size:12px;opacity:.7;"><b>skip reason:</b> ' + html.escape(str(d.get('skip_reason',''))) + '</div>') if d.get('skip_reason') else ''}
</div>"""

    metrics_html = _render_extended_metrics(fr)

    turns_html = []
    for t in d.get("turns", []):
        turns_html.append(_render_turn(t))

    model = d.get("model_name") or d["_model_dir"]
    variant = d.get("_variant", "default")
    head = (
        f'<header><div class="page-wrap">'
        f'<div class="site-title">L{d.get("level")} · problem {d.get("problem_id")}</div>'
        f'<div class="site-subtitle">{html.escape(model)} · {html.escape(variant)} · {html.escape(run_name)}</div>'
        f'<div class="header-links">'
        f'<a href="../../../index.html">← all variants</a>'
        f'<a href="../index.html">← {html.escape(variant)}</a>'
        f'<a href="../models/{_safe_filename(model)}.html">← {html.escape(model)}</a>'
        f'<span class="meta">generated {html.escape(generated_at)}</span>'
        f'</div></div></header>'
    )
    sitemap = (_render_sitemap(by_variant, link_prefix="../../../",
                               here=("trajectory", (variant, model)))
               if by_variant else "")
    body = head + sitemap + f'<div class="page-wrap">{tldr}{metrics_html}{"".join(turns_html)}</div>'
    return body


# ---------------------------------------------------------------------------
# Per-trajectory extended-metrics block
# ---------------------------------------------------------------------------

def _fmt(v: Any, suffix: str = "", prec: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return "—"
        # treat -1 as the eval.py "not measured" sentinel
        if isinstance(v, (int, float)) and v == -1:
            return "—"
        if isinstance(v, float):
            if abs(v) >= 1e6 or (0 < abs(v) < 1e-3):
                return f"{v:.{prec}e}{suffix}"
            return f"{v:.{prec}f}{suffix}"
        return f"{v}{suffix}"
    s = str(v)
    return s if s else "—"


def _ratio_class(v: float | None, *, higher_is_better: bool, neutral: float = 1.0) -> str:
    if v is None or v <= 0:
        return ""
    if higher_is_better:
        return "ratio-good" if v >= neutral else "ratio-bad"
    return "ratio-good" if v <= neutral else "ratio-bad"


def _metric(label: str, val: str, cls: str = "") -> str:
    return (f'<div class="metric"><div class="l">{html.escape(label)}</div>'
            f'<div class="v {cls}">{val}</div></div>')


def _section(title: str, source: str, cells: list[str]) -> str:
    if not cells:
        return ""
    return (f'<div class="metrics-section">'
            f'<div class="metrics-head"><span>{html.escape(title)}</span>'
            f'<span class="src">{html.escape(source)}</span></div>'
            f'<div class="metrics-grid">{"".join(cells)}</div></div>')


def _render_extended_metrics(fr: dict) -> str:
    """Render every extended eval metric stored on KernelExecResult."""
    if not fr:
        return ""

    sections: list[str] = []

    # Timing
    rt = fr.get("runtime", -1)
    rrt = fr.get("ref_runtime", -1)
    sp = (rrt / rt) if (rt and rrt and rt > 0 and rrt > 0) else None
    rstats = fr.get("runtime_stats") or {}
    refstats = fr.get("ref_runtime_stats") or {}
    timing_cells = [
        _metric("kernel runtime",     _fmt(rt,  " μs")),
        _metric("ref runtime",        _fmt(rrt, " μs")),
        _metric("speedup",            _fmt(sp, "x") if sp else "—",
                _ratio_class(sp, higher_is_better=True)),
        _metric("kernel mean / median / std",
                f"{_fmt(rstats.get('mean'),'')} / {_fmt(rstats.get('median'),'')} / {_fmt(rstats.get('std'),'')} μs"
                if rstats else "—"),
        _metric("ref mean / median / std",
                f"{_fmt(refstats.get('mean'),'')} / {_fmt(refstats.get('median'),'')} / {_fmt(refstats.get('std'),'')} μs"
                if refstats else "—"),
    ]
    # Translation mode
    src_rt = fr.get("source_runtime", -1)
    if src_rt and src_rt > 0:
        timing_cells.append(_metric("source runtime",
                                    f"{_fmt(src_rt,' μs')} ({fr.get('source_backend','?')})"))
        timing_cells.append(_metric("speedup vs source",
                                    _fmt(fr.get("speedup_vs_source"), "x"),
                                    _ratio_class(fr.get("speedup_vs_source"),
                                                 higher_is_better=True)))
    sections.append(_section("timing", "kernelbench.timing", timing_cells))

    # Numerical precision
    np_stats = fr.get("numerical_precision") or {}
    if np_stats:
        sections.append(_section("numerical precision", "eval_kernel_against_ref", [
            _metric("max abs error",  _fmt(np_stats.get("max_abs_error"))),
            _metric("mean abs error", _fmt(np_stats.get("mean_abs_error"))),
            _metric("max rel error",  _fmt(np_stats.get("max_rel_error"))),
            _metric("mean rel error", _fmt(np_stats.get("mean_rel_error"))),
        ]))

    # Memory
    mem = fr.get("memory_stats") or {}
    if mem and "error" not in mem:
        ratio = mem.get("memory_ratio")
        sections.append(_section("gpu memory", "extended_metrics.measure_memory", [
            _metric("kernel peak",  _fmt(mem.get("peak_memory_mb"), " MB")),
            _metric("ref peak",     _fmt(mem.get("ref_peak_memory_mb"), " MB")),
            _metric("kernel / ref", _fmt(ratio, "x"),
                    _ratio_class(ratio, higher_is_better=False)),
        ]))

    # Kernel launches / fusion
    kl = fr.get("kernel_launch_stats") or {}
    if kl and "error" not in kl and (kl.get("num_kernels", -1) or -1) > 0:
        fusion = kl.get("fusion_ratio")
        sections.append(_section("kernel launches / fusion", "torch.profiler", [
            _metric("kernel launches", _fmt(kl.get("num_kernels"), "", prec=0)),
            _metric("ref launches",    _fmt(kl.get("ref_num_kernels"), "", prec=0)),
            _metric("fusion ratio (ref / kernel)", _fmt(fusion, "x"),
                    _ratio_class(fusion, higher_is_better=True)),
            _metric("total cuda time",
                    _fmt(kl.get("total_cuda_time_us"), " μs", prec=0)),
        ]))

    # Energy
    en = fr.get("energy_stats") or {}
    if en and "error" not in en and (en.get("energy_per_run_mj", -1) or -1) > 0:
        eratio = en.get("energy_ratio")
        sections.append(_section("energy", "NVML (pynvml)", [
            _metric("kernel mJ / run", _fmt(en.get("energy_per_run_mj"))),
            _metric("ref mJ / run",    _fmt(en.get("ref_energy_per_run_mj"))),
            _metric("ratio (ref / kernel)", _fmt(eratio, "x"),
                    _ratio_class(eratio, higher_is_better=True)),
            _metric("avg power",       _fmt(en.get("avg_power_w"), " W")),
            _metric("ref avg power",   _fmt(en.get("ref_avg_power_w"), " W")),
        ]))

    # SOL
    sol = fr.get("sol_stats") or {}
    if sol and (sol.get("sol_score", -1) or -1) >= 0:
        sections.append(_section("SOL (speed-of-light)",
                                 sol.get("source", "—"), [
            _metric("SOL score (0–1)", _fmt(sol.get("sol_score"), "", prec=3),
                    _ratio_class(sol.get("sol_score"), higher_is_better=True, neutral=0.5)),
            _metric("dram util %",     _fmt(sol.get("dram_utilization_pct"), "%", prec=1)),
            _metric("compute util %",  _fmt(sol.get("compute_utilization_pct"), "%", prec=1)),
            _metric("bottleneck",      html.escape(str(sol.get("bottleneck", "—")))),
            _metric("dominant pipe",   html.escape(str(sol.get("dominant_pipe", "—")))),
            _metric("arithmetic intensity", _fmt(sol.get("arithmetic_intensity"))),
            _metric("ridge point",     _fmt(sol.get("ridge_point"))),
        ]))

    # Roofline / occupancy
    rl = fr.get("roofline_stats") or {}
    if rl and rl.get("source"):
        rl_cells = [
            _metric("dram bandwidth",   _fmt(rl.get("dram_bandwidth_gbs"), " GB/s")),
            _metric("dram util %",      _fmt(rl.get("dram_utilization_pct"), "%", prec=1)),
            _metric("fp32 tflops",      _fmt(rl.get("fp32_tflops"))),
            _metric("fp32 util %",      _fmt(rl.get("fp32_utilization_pct"), "%", prec=1)),
            _metric("fp16 tflops",      _fmt(rl.get("fp16_tflops"))),
            _metric("occupancy %",      _fmt(rl.get("occupancy_pct"), "%", prec=1)),
            _metric("bottleneck",       html.escape(str(rl.get("bottleneck", "—")))),
            _metric("dominant pipe %",  _fmt(rl.get("dominant_utilization_pct"), "%", prec=1)),
            _metric("arithmetic intensity", _fmt(rl.get("arithmetic_intensity"))),
            _metric("ridge point",      _fmt(rl.get("ridge_point"))),
            _metric("L1 hit %",         _fmt(rl.get("l1_hit_rate_pct"), "%", prec=1)),
            _metric("L2 hit %",         _fmt(rl.get("l2_hit_rate_pct"), "%", prec=1)),
            _metric("ld sectors / req", _fmt(rl.get("ld_sectors_per_request"))),
            _metric("st sectors / req", _fmt(rl.get("st_sectors_per_request"))),
            _metric("regs / thread",    _fmt(rl.get("registers_per_thread"), "", prec=0)),
            _metric("smem / block",     _fmt(rl.get("shared_mem_per_block"), " B", prec=0)),
            _metric("block size",       _fmt(rl.get("block_size"), "", prec=0)),
            _metric("peak bw",          _fmt(rl.get("peak_bw_gbs"), " GB/s")),
            _metric("peak fp32 tflops", _fmt(rl.get("peak_fp32_tflops"))),
            _metric("peak fp16 tflops", _fmt(rl.get("peak_fp16_tflops"))),
        ]
        sections.append(_section("roofline / occupancy", rl.get("source", "—"), rl_cells))

        # Warp stalls — top 5 reasons.
        ws = rl.get("warp_stalls") or {}
        if ws:
            top = sorted(ws.items(), key=lambda kv: -float(kv[1] or 0))[:5]
            sections.append(_section("warp stalls (top 5)", "nsight", [
                _metric(html.escape(str(name)), _fmt(val, "%", prec=1))
                for name, val in top
            ]))

    # Flags
    md = fr.get("metadata") or {}
    if md.get("excessive_speedup"):
        sections.append(_section("flags", "submit_kernel guard", [
            _metric("excessive speedup",
                    '<span class="ratio-bad">flagged</span>'),
        ]))

    if not sections:
        return ""
    return f'<div class="metrics">{"".join(sections)}</div>'


def _render_turn(t: dict) -> str:
    parts = [f'<div class="turn"><div class="turn-head">'
             f'<span class="turn-id">turn {t.get("turn_id","?")}</span>'
             f'<span>llm latency: {t.get("llm_latency_s",0):.1f}s</span>'
             f'{" · <b>FINAL</b>" if t.get("is_final") else ""}'
             f'</div>']

    # Render assistant response items: reasoning + text + function_calls
    for item in t.get("response", []) or []:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "reasoning":
            summary = ""
            for s in item.get("summary", []) or []:
                summary += s.get("text", "") + "\n"
            if summary.strip():
                parts.append(_details("reasoning", summary, cls="reasoning",
                                      open_=False))
        elif kind == "message":
            text = ""
            for c in item.get("content", []) or []:
                if isinstance(c, dict):
                    text += c.get("text", "") + "\n"
            if text.strip():
                parts.append(_block("assistant", text))
        elif kind == "function_call":
            name = item.get("name", "?")
            args_raw = item.get("arguments", "")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {"_raw": args_raw}
            parts.append(_render_function_call(name, args))

    # Tool call results (paired by name/order)
    for tc in t.get("tool_calls", []) or []:
        parts.append(_render_tool_result(tc))

    if t.get("feedback_to_model"):
        parts.append(_block("feedback", t["feedback_to_model"]))

    parts.append("</div>")
    return "".join(parts)


def _render_function_call(name: str, args: dict) -> str:
    # Truncate kernel_code so the page stays manageable
    args_pretty = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 800:
            v = v[:800] + f"\n... ({len(v)-800} more chars)"
        args_pretty[k] = v
    body = f'<span class="role-tag">call</span><b>{html.escape(name)}</b>\n' \
           + html.escape(json.dumps(args_pretty, indent=2))
    return _details(f"call → {name}", body, cls="tool-args", open_=False, raw=True)


def _render_tool_result(tc: dict) -> str:
    name = tc.get("tool_name", "?")
    out = tc.get("result_text", "")
    ok = bool(tc.get("success"))
    cls = "tool-out" if ok else "tool-out fail"
    body = f'<span class="role-tag">{("ok" if ok else "fail")}</span>' \
           f'<b>{html.escape(name)}</b>\n' + html.escape(out)
    return _details(f"result ← {name} ({'ok' if ok else 'fail'})", body,
                    cls=cls, open_=not ok, raw=True)


def _block(label: str, text: str, cls: str = "assistant") -> str:
    return (f'<div class="block {cls}"><span class="role-tag">{label}</span>'
            f'{html.escape(text)}</div>')


def _details(summary: str, body: str, cls: str = "", open_: bool = False,
             raw: bool = False) -> str:
    body_html = body if raw else html.escape(body)
    o = " open" if open_ else ""
    return (f'<details{o}><summary>{html.escape(summary)}</summary>'
            f'<div class="block {cls}">{body_html}</div></details>')


# ---------------------------------------------------------------------------
# Top-level: build_report
# ---------------------------------------------------------------------------

def build_report(run_dir: str) -> None:
    """Rebuild the static report site under {run_dir}/report/.

    Layout:
        report/index.html                       -- all-variants overview
        report/style.css                        -- shared styles
        report/v/{variant}/index.html           -- per-variant summary
        report/v/{variant}/models/{model}.html  -- model card grid
        report/v/{variant}/t/{traj_id}.html     -- one page per trajectory
    """
    import time as _time
    generated_at = _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime()) + " UTC"

    run_name = os.path.basename(os.path.normpath(run_dir))
    report_dir = os.path.join(run_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    # CSS
    with open(os.path.join(report_dir, "style.css"), "w") as f:
        f.write(_CSS)

    trajs = _load_all_trajectories(run_dir)

    # Group trajectories by variant.
    by_variant: dict[str, list[dict]] = {}
    for d in trajs:
        by_variant.setdefault(d["_variant"], []).append(d)

    # Top-level overview.
    top_body = _render_top_index(run_name, by_variant, generated_at)
    with open(os.path.join(report_dir, "index.html"), "w") as f:
        f.write(_page(f"{run_name} · sweep", top_body, css_path="style.css"))

    # One subreport per variant.
    for variant, vtrajs in by_variant.items():
        vdir = os.path.join(report_dir, "v", _safe_filename(variant))
        os.makedirs(os.path.join(vdir, "models"), exist_ok=True)
        os.makedirs(os.path.join(vdir, "t"), exist_ok=True)

        # variant index
        body = _render_variant_index(run_name, variant, vtrajs, generated_at,
                                     by_variant=by_variant)
        with open(os.path.join(vdir, "index.html"), "w") as f:
            f.write(_page(f"{variant} · {run_name}", body, css_path="../../style.css"))

        # per-model pages
        by_model: dict[str, list[dict]] = {}
        for d in vtrajs:
            by_model.setdefault(d.get("model_name") or d["_model_dir"], []).append(d)
        for model, items in by_model.items():
            body = _render_model_page(model, variant, items, run_name, generated_at,
                                      by_variant=by_variant)
            path = os.path.join(vdir, "models", f"{_safe_filename(model)}.html")
            with open(path, "w") as f:
                f.write(_page(f"{model} · {variant} · {run_name}", body,
                              css_path="../../../style.css"))

        # per-trajectory pages
        for d in vtrajs:
            body = _render_trajectory(d, run_name, generated_at, by_variant=by_variant)
            path = os.path.join(vdir, "t", f"{_trajectory_id(d)}.html")
            with open(path, "w") as f:
                f.write(_page(f"L{d.get('level')} P{d.get('problem_id')} · {variant}",
                              body, css_path="../../../style.css"))


def main():
    if len(sys.argv) < 2:
        print("usage: build_report.py <runs/{run_name} dir>")
        sys.exit(2)
    run_dir = sys.argv[1]
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(REPO_TOP_DIR, run_dir)
    build_report(run_dir)
    print(f"[build_report] wrote {os.path.join(run_dir, 'report', 'index.html')}")


if __name__ == "__main__":
    main()
