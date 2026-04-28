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
    speedups = [s for s in (_speedup(d) for d in trajs) if s is not None]
    avg_sp = sum(speedups) / len(speedups) if speedups else 0.0
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
    <div class="stat-val">{avg_sp:.2f}x</div></div>
</div>"""


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
        + _stat_row(all_trajs)
        + f'<div class="page-wrap"><div class="tool-grid">{"".join(cards)}</div></div>'
    )
    return body


def _render_variant_index(run_name: str, variant: str, trajs: list[dict],
                          generated_at: str) -> str:
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
    body = head + _stat_row(trajs) + f'<div class="page-wrap">{table}</div>'
    return body


def _render_model_page(model: str, variant: str, trajs: list[dict],
                       run_name: str, generated_at: str) -> str:
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
        f'<div class="page-wrap"><div class="tool-grid">{"".join(cards)}</div></div>'
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


def _render_trajectory(d: dict, run_name: str, generated_at: str) -> str:
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
    body = head + f'<div class="page-wrap">{tldr}{"".join(turns_html)}</div>'
    return body


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
        body = _render_variant_index(run_name, variant, vtrajs, generated_at)
        with open(os.path.join(vdir, "index.html"), "w") as f:
            f.write(_page(f"{variant} · {run_name}", body, css_path="../../style.css"))

        # per-model pages
        by_model: dict[str, list[dict]] = {}
        for d in vtrajs:
            by_model.setdefault(d.get("model_name") or d["_model_dir"], []).append(d)
        for model, items in by_model.items():
            body = _render_model_page(model, variant, items, run_name, generated_at)
            path = os.path.join(vdir, "models", f"{_safe_filename(model)}.html")
            with open(path, "w") as f:
                f.write(_page(f"{model} · {variant} · {run_name}", body,
                              css_path="../../../style.css"))

        # per-trajectory pages
        for d in vtrajs:
            body = _render_trajectory(d, run_name, generated_at)
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
