#!/usr/bin/env bash
# run_resume_sweeps.sh — resume the l2pop multi_turn_all crash, then run l3
# multi_turn_all. Both configs now point at the popcorn-centralus deployment
# of gpt-5.5 (POPCORN_CENTRAL_AZURE_KEY).
#
# Sequential, not parallel: profile_kernel (Nsight Compute) does not co-schedule
# on a single GPU, and the per-GPU perf-timing file lock is run-dir-scoped, so
# overlapping two multi_turn_all sweeps on the same 8xH100 box would corrupt
# both timing and ncu profiles. Each sweep already saturates the box on its own.
#
# l2pop multi_turn_all resumes from disk: trajectories with finished_at and
# outcome != "in_progress" are skipped, so the 17 already-completed runs from
# the prior crash are reused automatically.
#
#   tmux new -s sweeps -d 'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH; ./scripts/run_resume_sweeps.sh 2>&1 | tee resume_sweeps.log'
#   tmux new -s pub    -d 'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH; uv run python scripts/publish_to_gh_pages.py --watch 120'

set -euo pipefail
cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

CONFIGS=(
  configs/sweep.cuda_multi_turn_all_l2pop.toml
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

for c in "${CONFIGS[@]}"; do run "$c"; done

echo "all sweeps complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
