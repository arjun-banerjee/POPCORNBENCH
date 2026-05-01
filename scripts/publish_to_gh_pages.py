"""publish_to_gh_pages.py — sync sweep reports to a `gh-pages` branch.

Picks up every `runs/*/report/` directory, copies it into a git worktree on
the `gh-pages` branch under that run's name, regenerates a top-level
`index.html` linking to each, commits, and pushes.

Designed to coexist with a running sweep — the worktree lives outside the
working tree so it doesn't disturb your checkout. Use `--watch SECONDS` to
loop forever and republish on a fixed cadence.

Examples
--------
    # one-shot push of everything under runs/
    uv run python scripts/publish_to_gh_pages.py

    # loop every 5 minutes (run inside tmux while the sweep is running)
    uv run python scripts/publish_to_gh_pages.py --watch 300

    # restrict to a single run
    uv run python scripts/publish_to_gh_pages.py --run multi_turn_default

GitHub side (one-time):
    Settings → Pages → Source: Deploy from a branch
                       Branch: gh-pages   Folder: / (root)
    URL:   https://<user>.github.io/<repo>/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRANCH = "gh-pages"
DEFAULT_WORKTREE = REPO_ROOT.parent / f"{REPO_ROOT.name}-gh-pages"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True,
         capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command for observability."""
    print(f"  $ {' '.join(cmd)}" + (f"   (cwd={cwd})" if cwd else ""))
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=check,
        capture_output=capture, text=True,
    )


def _branch_exists_remote(branch: str) -> bool:
    res = _run(["git", "ls-remote", "--heads", "origin", branch],
               cwd=REPO_ROOT, capture=True)
    return bool(res.stdout.strip())


def _branch_exists_local(branch: str) -> bool:
    res = _run(["git", "rev-parse", "--verify", "--quiet", branch],
               cwd=REPO_ROOT, check=False, capture=True)
    return res.returncode == 0


def _ensure_gh_pages_branch(branch: str) -> None:
    """Create an orphan branch with an initial commit if neither remote nor
    local has it. Safe to re-run."""
    if _branch_exists_remote(branch):
        print(f"[publish] remote branch '{branch}' already exists.")
        # Make sure we have it locally too.
        if not _branch_exists_local(branch):
            _run(["git", "fetch", "origin", f"{branch}:{branch}"], cwd=REPO_ROOT)
        return
    if _branch_exists_local(branch):
        print(f"[publish] local branch '{branch}' exists, pushing to origin.")
        _run(["git", "push", "-u", "origin", branch], cwd=REPO_ROOT)
        return

    # Create orphan branch via a temporary worktree to avoid disturbing main.
    print(f"[publish] creating orphan branch '{branch}' …")
    tmp_dir = REPO_ROOT.parent / f"{REPO_ROOT.name}-bootstrap-ghpages"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    _run(["git", "worktree", "add", "--detach", str(tmp_dir)], cwd=REPO_ROOT)
    try:
        _run(["git", "checkout", "--orphan", branch], cwd=tmp_dir)
        _run(["git", "rm", "-rf", "."], cwd=tmp_dir, check=False)
        # Wipe any leftover working-tree files.
        for child in tmp_dir.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        # .nojekyll so GitHub Pages serves files prefixed with `_` correctly.
        (tmp_dir / ".nojekyll").write_text("")
        (tmp_dir / "index.html").write_text(_placeholder_html())
        _run(["git", "add", ".nojekyll", "index.html"], cwd=tmp_dir)
        _run(["git", "commit", "-m", "Initialize gh-pages"], cwd=tmp_dir)
        _run(["git", "push", "-u", "origin", branch], cwd=tmp_dir)
    finally:
        _run(["git", "worktree", "remove", "--force", str(tmp_dir)],
             cwd=REPO_ROOT, check=False)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _placeholder_html() -> str:
    return ("<!DOCTYPE html><html><head><meta charset=utf-8>"
            "<title>PopcornBench reports</title></head>"
            "<body><h1>PopcornBench reports</h1>"
            "<p>No reports published yet — the sweep is still warming up.</p>"
            "</body></html>")


def _find_reports(runs_dir: Path, only: str | None) -> list[tuple[str, Path]]:
    """Return [(run_name, report_dir)] for every runs/*/report/index.html."""
    if not runs_dir.exists():
        return []
    found: list[tuple[str, Path]] = []
    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        if only and run_path.name != only:
            continue
        report_dir = run_path / "report"
        index = report_dir / "index.html"
        if index.exists():
            found.append((run_path.name, report_dir))
    return found


def _copy_report(src: Path, dst: Path) -> None:
    """Mirror src into dst, replacing whatever was there before."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _ensure_worktree(branch: str, worktree: Path) -> None:
    """Make sure `worktree` is a worktree of `branch`, freshly synced from origin."""
    if worktree.exists() and not (worktree / ".git").exists():
        # Stale dir from a failed run — wipe it.
        shutil.rmtree(worktree)
    if not worktree.exists():
        _run(["git", "fetch", "origin", branch], cwd=REPO_ROOT, check=False)
        _run(["git", "worktree", "add", str(worktree), branch], cwd=REPO_ROOT)
    # Sync to origin's tip in case another publisher pushed.
    _run(["git", "fetch", "origin", branch], cwd=worktree, check=False)
    _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=worktree, check=False)


# ---------------------------------------------------------------------------
# Theme: popcorn-box colors. Used by every generated page.
#   --bg:   white page background
#   --fg:   body text (near-black)
#   --pri:  primary brand red (headings, borders)
#   --acc:  brighter red, used for links/hover
#   --soft: warm yellow accent (panels, hover backgrounds)
# ---------------------------------------------------------------------------
_TYPEKIT_LINK = '<link rel="stylesheet" href="https://use.typekit.net/bsk0vur.css">'

_COMMON_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#ffffff; --fg:#1a1a1a; --pri:#901010; --acc:#ab1313;
  --soft:#f8de8d; --soft08:rgba(248,222,141,.4); --pri15:rgba(144,16,16,.15);
  --pri08:rgba(144,16,16,.08); --pri35:rgba(144,16,16,.35); --t:80ms ease;
}
html,body{background:var(--bg);color:var(--fg);
  font-family:"calling-code",ui-monospace,Menlo,monospace;font-size:15px;line-height:1.6}
a{color:var(--acc);text-decoration:none}
a:hover{opacity:.65}
header{border-bottom:1px solid var(--pri);padding-bottom:0;background:var(--bg)}
header .wrap{padding:48px 24px 28px}
h1{font-family:"neue-kabel",sans-serif;font-weight:900;font-style:normal;font-size:64px;
  letter-spacing:-.025em;color:var(--pri);line-height:.95;margin-bottom:10px}
.sub{font-family:"calling-code",sans-serif;font-style:italic;font-size:14px;
  color:var(--pri);opacity:.55;margin-top:4px}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
main.wrap{padding-top:32px;padding-bottom:56px}
.section-head{font-size:11px;font-style:italic;letter-spacing:.08em;color:var(--pri);
  text-transform:uppercase;margin-bottom:12px;opacity:.65}
nav.topnav{display:flex;gap:16px;margin-top:14px;font-size:13px;font-style:italic}
nav.topnav a{color:var(--pri);opacity:.55;border-bottom:1px solid transparent;padding-bottom:1px}
nav.topnav a:hover{opacity:1}
nav.topnav a.current{opacity:1;border-bottom:2px solid var(--pri);font-style:normal;font-weight:600}
"""


_TOPNAV_ITEMS = [
    ("Home", "index.html"),
    ("Experiments", "experiments.html"),
    ("Docs", "docs.html"),
    ("Prompts", "prompts.html"),
]


def _topnav(current: str) -> str:
    """Same nav on every page; the current page is marked with .current."""
    out = []
    for label, href in _TOPNAV_ITEMS:
        cls = ' class="current"' if label.lower() == current.lower() else ""
        out.append(f'<a href="{href}"{cls}>{label}</a>')
    return f'<nav class="topnav">{"".join(out)}</nav>'


def _page(title: str, body: str, *, refresh: bool = False, extra_css: str = "") -> str:
    refresh_meta = '<meta http-equiv=refresh content=120>' if refresh else ""
    return (
        f"<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>{refresh_meta}"
        f"{_TYPEKIT_LINK}"
        f"<style>{_COMMON_CSS}{extra_css}</style></head>"
        f"<body>{body}</body></html>"
    )


def _build_homepage(worktree: Path, n_reports: int) -> None:
    when = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    blurb = (
        "PopcornBench is an extension of KernelBench with a multi-model "
        "agentic sweep harness for evaluating how well LLMs can write fast, "
        "numerically equivalent CUDA / Triton / Tilelang kernels against a "
        "PyTorch reference. Each run pairs a model with a level and problem "
        "set, drives a tool-using agent loop with compile / correctness / "
        "submit tools, and records per-problem speedups, fusion ratios, "
        "energy, and roofline metrics."
    )
    cards = (
        f'<a class="hero-card" href="experiments.html">'
        f'<div class="hero-name">Experiments</div>'
        f'<div class="hero-hint">{n_reports} sweep run{"s" if n_reports != 1 else ""}, per-model and per-problem reports</div>'
        f'<div class="hero-arrow">→</div></a>'
        f'<a class="hero-card" href="docs.html">'
        f'<div class="hero-name">Docs</div>'
        f'<div class="hero-hint">getting started, adding models, AlphaEvolve</div>'
        f'<div class="hero-arrow">→</div></a>'
        f'<a class="hero-card" href="prompts.html">'
        f'<div class="hero-name">Prompts</div>'
        f'<div class="hero-hint">system, user, and tool prompts the agent receives</div>'
        f'<div class="hero-arrow">→</div></a>'
    )

    plan = """
<h2>Plan</h2>
<p>We score each correct kernel on three signals and compare submissions on the Pareto frontier across all three:</p>
<ul class="signals">
  <li><b>Runtime.</b> Wall-clock per call relative to the PyTorch reference. Reported as speedup = ref_runtime / runtime.</li>
  <li><b>SOL.</b> max(dram_utilization, compute_utilization) / 100 from Nsight counters, bounded in [0, 1]. Heuristic fallback when ncu is unavailable.</li>
  <li><b>Energy.</b> Joules per call from NVML (energy_per_run_mj). Meaningful only when per-call runtime exceeds the NVML sampling window (~10 ms), so reported on L3 and L4 only.</li>
</ul>
<p>A kernel is considered to dominate another only if it is at least as good on all three signals and strictly better on one. Models are then ranked by the size of their Pareto-optimal subset within each (level, variant).</p>

<h3>Sweeps that feed the figures</h3>
<table class="plan-table">
<tr><th>Run</th><th>Coverage</th><th>Models</th></tr>
<tr><td><code>full_l12</code></td><td>L1, L2, original + popcorn, all problems</td><td>gpt-5.5, FW-GLM-5-1, Llama-4-Maverick, Kimi-K2.6</td></tr>
<tr><td><code>full_l34</code></td><td>L3, L4, original + popcorn, all problems</td><td>same four</td></tr>
<tr><td><code>ae_focus_popcorn</code></td><td>L1-L4 popcorn, problems 1, 2, 10</td><td>AlphaEvolve (Gemini 3.0 mixture)</td></tr>
</table>

<h3>Planned figures</h3>
<ol class="plan-figs">
  <li><b>Pareto frontier in (speedup, SOL).</b> One panel per level. Source: <code>full_l12</code> + <code>full_l34</code>.</li>
  <li><b>Pareto frontier in (speedup, 1 / energy_ratio).</b> L3 and L4 only. Source: <code>full_l34</code>.</li>
  <li><b>Speedup vs SOL scatter, all problems.</b> One dot per (model, problem), colored by level, faceted by variant. Source: <code>full_l12</code> + <code>full_l34</code>.</li>
  <li><b>Correctness rate stacked by tier on L2-popcorn.</b> Source: existing <code>cuda_l2pop_*</code> sweeps.</li>
  <li><b>SOL distribution by tier on L2-popcorn.</b> Source: existing L2-pop sweeps after the SOL clamp fix.</li>
  <li><b>AlphaEvolve speedup vs best non-AE speedup, per problem.</b> Source: <code>ae_focus_popcorn</code> joined to <code>full_*</code>.</li>
  <li><b>Failure-mode breakdown per (model, level).</b> Stacked correct / wrong / compile_fail / timeout / error. Source: <code>full_l12</code> + <code>full_l34</code>.</li>
</ol>
"""

    body = (
        f'<header><div class="wrap"><h1>PopcornBench</h1>'
        f'<div class="sub">multi-model kernel-benchmark harness, last published {html.escape(when)}</div>'
        f'{_topnav("Home")}'
        f'</div></header>'
        f'<main class="wrap">'
        f'<p class="blurb">{blurb}</p>'
        f'<div class="hero-grid">{cards}</div>'
        f'{plan}'
        f'</main>'
    )
    extra = """
.blurb{max-width:720px;margin:8px 0 30px;font-size:14.5px;line-height:1.7}
.hero-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:8px}
.hero-card{display:block;padding:22px 24px;border:2px solid var(--pri);background:var(--bg);
  color:var(--fg);text-decoration:none;transition:background var(--t),color var(--t);position:relative}
.hero-card:hover{background:var(--soft);text-decoration:none}
.hero-name{font-family:"neue-kabel",sans-serif;font-weight:700;font-style:normal;font-size:30px;color:var(--pri);letter-spacing:-.01em}
.hero-hint{font-size:12.5px;opacity:.75;margin-top:6px}
.hero-arrow{position:absolute;top:18px;right:22px;font-size:18px;color:var(--pri)}
main.wrap h2{font-family:"neue-kabel",sans-serif;font-weight:900;font-size:36px;
  color:var(--pri);margin:42px 0 12px;letter-spacing:-.02em}
main.wrap h3{font-family:"neue-kabel",sans-serif;font-weight:700;font-size:18px;
  color:var(--pri);margin:24px 0 10px;letter-spacing:-.005em}
main.wrap p{margin:8px 0}
ul.signals,ol.plan-figs{margin:8px 0 14px 20px}
ul.signals li,ol.plan-figs li{margin:6px 0}
table.plan-table{border-collapse:collapse;margin:10px 0 18px;width:100%;font-size:13px}
table.plan-table th,table.plan-table td{padding:8px 10px;border-bottom:1px solid var(--pri15);
  text-align:left;vertical-align:top}
table.plan-table th{color:var(--pri);font-weight:600}
code{background:var(--soft08);padding:1px 5px;border-radius:3px;font-size:13px;color:var(--pri)}
"""
    (worktree / "index.html").write_text(_page("PopcornBench", body, refresh=True, extra_css=extra))
    (worktree / ".nojekyll").write_text("")


_VARIANT_COLS = [
    ("single_turn", "single turn"),
    ("multi_turn_default", "multi turn (default tools)"),
    ("multi_turn_all", "multi turn (all tools)"),
]


def _classify_run(name: str) -> tuple[str | None, str | None]:
    """Bucket a run name into a (row, column) for the comparison matrix.

    Recognized rows are arbitrary level/variant tags — anything between
    "cuda_" and the trailing variant suffix. Unrecognized names are bucketed
    as (None, None) and shown only in the flat list.
    """
    for col_key, _ in _VARIANT_COLS:
        suffix = "_" + col_key
        if name.endswith(suffix):
            row = name[len("cuda_"):-len(suffix)] if name.startswith("cuda_") else name[:-len(suffix)]
            return row, col_key
    return None, None


def _build_experiments(worktree: Path, reports: list[tuple[str, Path]]) -> None:
    """Render the experiments page: comparison matrix + flat list."""
    when = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    by_cell: dict[tuple[str, str], list[str]] = {}
    rows_seen: list[str] = []
    flat = sorted(name for name, _ in reports)
    for name in flat:
        row, col = _classify_run(name)
        if row is None or col is None:
            continue
        if row not in rows_seen:
            rows_seen.append(row)
        by_cell.setdefault((row, col), []).append(name)
    rows_seen.sort()

    matrix = ""
    if rows_seen:
        head = "<th></th>" + "".join(f"<th>{html.escape(label)}</th>" for _, label in _VARIANT_COLS)
        body_rows = []
        for row in rows_seen:
            cells = [f'<th class="row-head">{html.escape(row)}</th>']
            for col_key, _ in _VARIANT_COLS:
                names = by_cell.get((row, col_key), [])
                if not names:
                    cells.append('<td class="cell cell-empty">—</td>')
                else:
                    inner = "".join(
                        f'<a href="{html.escape(n)}/index.html">'
                        f'<span class="cell-name">{html.escape(n)}</span>'
                        f'<span class="cell-hint">open →</span></a>'
                        for n in names
                    )
                    cells.append(f'<td class="cell">{inner}</td>')
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        matrix = (
            '<div class="section-head">Compare runs</div>'
            f'<table class="matrix"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table>'
        )

    flat_cards = "".join(
        f'<a class="flat-card" href="{html.escape(n)}/index.html">'
        f'<span class="flat-name">{html.escape(n)}</span>'
        f'<span class="flat-hint">→</span></a>'
        for n in flat
    ) or '<div class="cell-empty" style="padding:24px">No reports yet.</div>'

    body = (
        f'<header><div class="wrap"><h1>Experiments</h1>'
        f'<div class="sub">{len(flat)} run{"s" if len(flat) != 1 else ""}, updated {html.escape(when)}</div>'
        f'{_topnav("Experiments")}'
        f'</div></header>'
        f'<main class="wrap">'
        f'{matrix}'
        f'<div class="section-head" style="margin-top:32px">All runs</div>'
        f'<div class="flat-grid">{flat_cards}</div>'
        f'</main>'
    )
    extra = """
.matrix{width:100%;border-collapse:collapse;border:1px solid var(--pri);margin-top:6px}
.matrix th,.matrix td{padding:10px 14px;border:1px solid var(--pri15);font-size:13px;vertical-align:top}
.matrix thead th{background:var(--pri);color:var(--bg);font-weight:600;font-size:12px;text-align:center;border-color:var(--pri)}
.matrix .row-head{background:var(--soft08);font-weight:600;font-size:13px;white-space:nowrap;border-color:var(--pri35);color:var(--pri)}
.cell{background:var(--bg)}
.cell a{display:flex;flex-direction:column;gap:4px;padding:6px 8px;margin:-6px -8px;border:1px solid transparent;transition:background var(--t),border-color var(--t);color:var(--fg)}
.cell a:hover{background:var(--soft);border-color:var(--pri35);text-decoration:none}
.cell-name{font-size:12px;font-weight:500;word-break:break-word}
.cell-hint{font-size:11px;opacity:.6;color:var(--acc)}
.cell-empty{text-align:center;opacity:.45;background:var(--pri08)}
.flat-grid{display:flex;flex-wrap:wrap;gap:1px;background:var(--pri);border:1px solid var(--pri)}
.flat-card{background:var(--bg);padding:10px 16px;display:flex;align-items:center;justify-content:space-between;gap:12px;transition:background var(--t),color var(--t);min-width:260px;flex:1;color:var(--fg)}
.flat-card:hover{background:var(--soft);text-decoration:none}
.flat-name{font-size:13px;word-break:break-word}
.flat-hint{font-size:12px;color:var(--acc);flex-shrink:0}
"""
    (worktree / "experiments.html").write_text(
        _page("PopcornBench experiments", body, refresh=True, extra_css=extra)
    )


def _build_index(worktree: Path, reports: list[tuple[str, Path]]) -> None:
    """Generate every static page on the site."""
    _build_homepage(worktree, n_reports=len(reports))
    _build_experiments(worktree, reports)
    _build_docs(worktree)
    _build_prompts(worktree)


def _build_docs(worktree: Path) -> None:
    """Write the docs.html page (getting started + adding models + AlphaEvolve)."""
    docs_src = REPO_ROOT / "scripts" / "_docs_page.html"
    if docs_src.exists():
        (worktree / "docs.html").write_text(docs_src.read_text())


def _build_prompts(worktree: Path) -> None:
    """Render prompts.html showing the system prompt, user message, and tools.

    Subprocesses scripts/_dump_prompts.py so we don't have to import torch
    into the publisher process.
    """
    dumper = REPO_ROOT / "scripts" / "_dump_prompts.py"
    if not dumper.exists():
        return
    try:
        out = subprocess.run(
            ["uv", "run", "python", str(dumper)],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
        )
        data = json.loads(out.stdout)
    except Exception as e:
        print(f"[publish] could not dump prompts: {e}")
        return

    when = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _section(title: str, body: str, *, open_first: bool = False) -> str:
        attr = " open" if open_first else ""
        return (
            f'<details class="prompt-section"{attr}>'
            f'<summary>{html.escape(title)}</summary>'
            f'<div class="prompt-body">{body}</div>'
            f'</details>'
        )

    def _pre(text: str) -> str:
        return f'<pre class="prompt-pre">{html.escape(text)}</pre>'

    sections = [
        _section(
            "System prompt (default tool set)",
            _pre(data["system_default"]),
            open_first=True,
        ),
        _section(
            "System prompt (all tools, including profile / disassemble / ert)",
            _pre(data["system_all_tools"]),
        ),
        _section(
            "First user message (problem statement)",
            _pre(data["problem_message"]),
        ),
        _section(
            "Turn-warning message (injected when budget runs low)",
            _pre(data["turn_warning"]),
        ),
    ]

    tool_blocks = []
    for t in data["tools"]:
        head = (
            f'<code>{html.escape(t["name"])}</code>'
            + (' <span class="default-pill">default</span>' if t["in_default_set"] else '')
        )
        body = (
            f'<div class="tool-desc">{html.escape(t["description"])}</div>'
            f'<div class="tool-schema-label">input_schema</div>'
            f'<pre class="prompt-pre">{html.escape(json.dumps(t["input_schema"], indent=2))}</pre>'
        )
        tool_blocks.append(
            f'<details class="tool-section">'
            f'<summary>{head}</summary>'
            f'<div class="prompt-body">{body}</div>'
            f'</details>'
        )

    body = (
        f'<header><div class="wrap"><h1>Prompts</h1>'
        f'<div class="sub">Every prompt the agent receives. Updated {html.escape(when)}.</div>'
        f'{_topnav("Prompts")}'
        f'</div></header>'
        f'<main class="wrap">'
        f'<p class="intro">Generated from <code>build_system_prompt</code>, <code>build_problem_message</code>, and the tool registry. Click any section to expand.</p>'
        f'<h2>Conversation prompts</h2>'
        f'{"".join(sections)}'
        f'<h2>Tool descriptions</h2>'
        f'<p class="intro">Each entry below is one function declaration sent to the model. <code>default</code> means the tool is included when <code>tools = "default"</code> in the sweep TOML.</p>'
        f'{"".join(tool_blocks)}'
        f'</main>'
    )
    extra = """
.intro{max-width:720px;margin:6px 0 12px;font-size:13.5px;line-height:1.6;opacity:.85}
main.wrap h2{font-family:"neue-kabel",sans-serif;font-weight:900;font-size:32px;
  color:var(--pri);margin:34px 0 14px;letter-spacing:-.02em}
.prompt-section, .tool-section{
  border:1px solid var(--pri15);background:var(--bg);margin:6px 0;
}
.prompt-section > summary, .tool-section > summary{
  cursor:pointer;padding:11px 14px;font-size:14px;
  border-bottom:1px solid transparent;list-style:none;
  display:flex;align-items:center;gap:10px;
}
.prompt-section[open] > summary, .tool-section[open] > summary{
  border-bottom:1px solid var(--pri15);background:var(--soft08);
}
.prompt-section > summary::before, .tool-section > summary::before{
  content:"▸ ";color:var(--pri);opacity:.7;
}
.prompt-section[open] > summary::before, .tool-section[open] > summary::before{
  content:"▾ ";
}
.prompt-section > summary::-webkit-details-marker,
.tool-section > summary::-webkit-details-marker{display:none}
.prompt-body{padding:12px 16px}
.prompt-pre{
  background:var(--soft08);padding:12px 14px;font-size:12.5px;line-height:1.55;
  white-space:pre-wrap;word-break:break-word;
  border-left:3px solid var(--pri);overflow-x:auto;
}
.tool-section > summary code{
  font-size:14px;font-weight:600;color:var(--pri);background:none;padding:0;
}
.default-pill{
  font-size:10px;border:1px solid var(--pri);color:var(--pri);
  padding:1px 6px;border-radius:2px;letter-spacing:.04em;
}
.tool-desc{margin-bottom:10px;line-height:1.55;font-size:13px}
.tool-schema-label{
  font-size:10px;letter-spacing:.08em;text-transform:uppercase;opacity:.6;
  margin:6px 0 4px;
}
"""
    (worktree / "prompts.html").write_text(
        _page("PopcornBench prompts", body, refresh=False, extra_css=extra)
    )


def _rebuild_reports(reports: list[tuple[str, Path]]) -> None:
    """Re-run build_report.py against every detected run before copying.

    Why subprocess instead of importing build_report directly: the running
    sweep already has build_report imported into its own process and Python
    caches that import, so updates from `git pull` aren't picked up there.
    Running build_report as a fresh subprocess from the publisher guarantees
    every cycle uses the latest code on disk — no sweep restart required.
    """
    builder = REPO_ROOT / "scripts" / "build_report.py"
    for name, report_dir in reports:
        run_dir = report_dir.parent  # runs/{name}/
        try:
            subprocess.run(
                ["uv", "run", "python", str(builder), str(run_dir)],
                cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "").strip()[-500:]
            print(f"[publish] build_report failed for {name}: {tail}")


def _has_changes(worktree: Path) -> bool:
    res = _run(["git", "status", "--porcelain"], cwd=worktree, capture=True)
    return bool(res.stdout.strip())


def publish_once(*, runs_dir: Path, branch: str, worktree: Path,
                 only: str | None) -> bool:
    """Returns True when something was pushed, False when there were no changes."""
    print(f"\n[publish] sync at {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    _ensure_gh_pages_branch(branch)
    _ensure_worktree(branch, worktree)

    reports = _find_reports(runs_dir, only)
    if not reports:
        print(f"[publish] no reports found under {runs_dir} "
              f"(looked for runs/*/report/index.html)")
        return False
    print(f"[publish] found {len(reports)} report(s): "
          f"{', '.join(n for n, _ in reports)}")

    # Always rebuild reports first so any code changes (git pull) take effect
    # immediately — the running sweep's in-process build_report cache misses
    # updates, but the publisher's subprocess invocation does not.
    print("[publish] rebuilding reports from latest build_report.py …")
    _rebuild_reports(reports)

    for name, src in reports:
        dst = worktree / name
        _copy_report(src, dst)

    _build_index(worktree, reports)

    if not _has_changes(worktree):
        print("[publish] no changes to commit.")
        return False

    _run(["git", "add", "-A"], cwd=worktree)
    msg = (f"publish reports {_dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} "
           f"({len(reports)} run{'s' if len(reports) != 1 else ''})")
    _run(["git", "commit", "-m", msg], cwd=worktree)
    _run(["git", "push", "origin", branch], cwd=worktree)
    print(f"[publish] pushed {msg}")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"),
                   help="Directory containing runs/{name}/report/ subdirs.")
    p.add_argument("--branch", default=DEFAULT_BRANCH,
                   help="GitHub Pages branch (default: gh-pages).")
    p.add_argument("--worktree", default=str(DEFAULT_WORKTREE),
                   help="Path to the git worktree used for publishing.")
    p.add_argument("--run", default=None,
                   help="Restrict to a single run name (default: publish all).")
    p.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                   help="Loop forever, republishing every SECONDS. 0 = one-shot.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    worktree = Path(args.worktree).resolve()

    while True:
        try:
            publish_once(runs_dir=runs_dir, branch=args.branch,
                         worktree=worktree, only=args.run)
        except subprocess.CalledProcessError as e:
            print(f"[publish] command failed: {e}", file=sys.stderr)
            # don't exit the watch loop on a transient git error
            if not args.watch:
                return 1
        except Exception as e:
            print(f"[publish] error: {e}", file=sys.stderr)
            if not args.watch:
                return 1

        if args.watch <= 0:
            return 0
        print(f"[publish] sleeping {args.watch}s")
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
