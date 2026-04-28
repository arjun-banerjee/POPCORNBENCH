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


def _build_index(worktree: Path, reports: list[tuple[str, Path]]) -> None:
    """Write a top-level index.html listing every published run."""
    when = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    cards = []
    for name, _ in sorted(reports, key=lambda x: x[0]):
        cards.append(
            f'<a class="card" href="{html.escape(name)}/index.html">'
            f'<div class="name">{html.escape(name)}</div>'
            f'<div class="hint">open report →</div></a>'
        )
    body = (
        f'<header><h1>PopcornBench sweep reports</h1>'
        f'<div class="meta">last published {html.escape(when)}</div></header>'
        f'<div class="grid">{"".join(cards) or "<p>No reports yet.</p>"}</div>'
    )
    css = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--g:#00553A;--w:#fff;--g08:rgba(0,85,58,.08)}
html,body{background:var(--w);color:var(--g);
  font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:14px;line-height:1.55}
header{padding:36px 24px 24px;border-bottom:1px solid var(--g);max-width:900px;margin:0 auto}
h1{font-weight:400;font-style:italic;font-size:32px;letter-spacing:-.02em}
.meta{font-size:12px;font-style:italic;opacity:.45;margin-top:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:1px;background:var(--g);max-width:900px;margin:24px auto;padding-bottom:1px}
.card{background:var(--w);padding:18px 20px;text-decoration:none;color:var(--g);
  display:flex;flex-direction:column;gap:6px;transition:background 80ms ease,color 80ms ease}
.card:hover{background:var(--g);color:var(--w)}
.name{font-size:15px}
.hint{font-size:11px;font-style:italic;opacity:.55}
"""
    page = (f"<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>PopcornBench reports</title>"
            f"<meta http-equiv=refresh content=120>"
            f"<style>{css}</style></head><body>{body}</body></html>")
    (worktree / "index.html").write_text(page)
    (worktree / ".nojekyll").write_text("")


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
