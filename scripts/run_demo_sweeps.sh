#!/usr/bin/env bash
# run_demo_sweeps.sh — phased demo sweep for the morning presentation.
#
# Phase B first (small, ~2.5 hr) so we have a guaranteed-complete sweep early.
# Phase A (L3, both variants, ~6 hr) runs after — partial results still publish
# every 2 min via publish_to_gh_pages.py.
#
#   tmux new -s sweeps -d 'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH; ./scripts/run_demo_sweeps.sh 2>&1 | tee sweep.log'
#   tmux new -s pub    -d 'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH; uv run python scripts/publish_to_gh_pages.py --watch 120'

set -euo pipefail
cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

# Phase B: L2 popcorn only — fast, guarantees something to present.
PHASE_B=(
  configs/sweep.cuda_single_turn_l2pop.toml
  configs/sweep.cuda_multi_turn_default_l2pop.toml
  configs/sweep.cuda_multi_turn_all_l2pop.toml
)

# Phase A: L3 both variants — heavy, may not finish; partials publish anyway.
PHASE_A=(
  configs/sweep.cuda_single_turn_l3.toml
  configs/sweep.cuda_multi_turn_default_l3.toml
  configs/sweep.cuda_multi_turn_all_l3.toml
)

run() {
  local cfg="$1"
  echo
  echo "================================================================"
  echo "▶ $(date -u +%Y-%m-%dT%H:%M:%SZ)  $cfg"
  echo "================================================================"
  uv run python scripts/run_sweep.py "$cfg"
}

echo "===== PHASE B (L2 popcorn) ====="
for c in "${PHASE_B[@]}"; do run "$c"; done

echo "===== PHASE A (L3 both variants) ====="
for c in "${PHASE_A[@]}"; do run "$c"; done

echo "all phases complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
