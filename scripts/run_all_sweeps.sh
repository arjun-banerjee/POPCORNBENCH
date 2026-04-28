#!/usr/bin/env bash
# run_all_sweeps.sh — run the three sweep configs in sequence.
#
# Each config wants the full 8xH100 box, so they run sequentially. Reports
# land under runs/{single_turn,multi_turn_default,multi_turn_all}/ and are
# republished to GitHub Pages every five minutes by publish_to_gh_pages.py.
#
# Recommended layout (run inside tmux). The LD_LIBRARY_PATH export is the
# conda-libstdc++ fix needed for torch / nvcc to find the right runtime:
#
#   tmux new -s sweeps  'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH; ./scripts/run_all_sweeps.sh'
#   tmux new -s pub     'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH; uv run python scripts/publish_to_gh_pages.py --watch 300'
#
# Live local report (per-sweep, while it's running):
#   ssh -L 8765:localhost:8765 user@host    # then http://localhost:8765
#
# Mobile / shareable (always up to date):
#   https://<github-user>.github.io/<repo>/

set -euo pipefail
cd "$(dirname "$0")/.."

# Conda libstdc++ fix — torch / nvcc otherwise pick up the wrong runtime.
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

CONFIGS=(
  configs/sweep.cuda_single_turn.toml
  configs/sweep.cuda_multi_turn_default.toml
  configs/sweep.cuda_multi_turn_all.toml
)

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "================================================================"
  echo "▶ $(date -u +%Y-%m-%dT%H:%M:%SZ)  starting sweep: $cfg"
  echo "================================================================"
  uv run python scripts/run_sweep.py "$cfg"
done

echo
echo "all sweeps complete — final report rebuild:"
for cfg in "${CONFIGS[@]}"; do
  name=$(uv run --no-project --with tomli python -c "
import tomli, sys
with open('$cfg','rb') as f: print(tomli.load(f)['run']['name'])
")
  uv run python scripts/build_report.py "runs/$name" || true
done
echo "done."
