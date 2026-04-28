"""
Standalone profiling worker for nsight-python.

This script exists so that when nsight-python relaunches the "application
script" under ncu, it relaunches THIS lightweight worker — not the batch
agent driver (run_agent_batch.py), which would recursively restart the
entire agent loop.

Protocol (parent → worker):
    1. Parent writes a JSON request to a temp file.
    2. Parent invokes:  python scripts/_profile_worker.py <request_json_path>
    3. Worker loads the request, runs profiling, prints a JSON response
       (a dict of metric_name → float|null) to stdout, then exits.

If profiling fails, the worker prints a JSON object with an "error" key
and exits with code 1.
"""

import json
import sys
import os

import torch


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: _profile_worker.py <request.json>"}))
        sys.exit(1)

    request_path = sys.argv[1]
    with open(request_path) as f:
        req = json.load(f)

    custom_model_src = req["custom_model_src"]
    ref_model_src = req.get("ref_model_src")
    metrics = req.get("metrics")
    num_trials = req.get("num_trials", 1)
    seed = req.get("seed", 42)
    device_index = req.get("device_index", 0)
    backend = req.get("backend", "cuda")
    precision = req.get("precision", "fp32")
    build_dir = req.get("build_dir")
    verbose = req.get("verbose", False)

    from kernelbench.eval import get_torch_dtype_from_string

    torch_precision = get_torch_dtype_from_string(precision)
    device = torch.device(f"cuda:{device_index}")

    from kernelbench.profile import profile_kernelbench_model_with_nsight

    result = profile_kernelbench_model_with_nsight(
        custom_model_src=custom_model_src,
        ref_model_src=ref_model_src,
        metrics=metrics,
        num_trials=num_trials,
        seed=seed,
        device=device,
        backend=backend,
        precision=torch_precision,
        build_dir=build_dir,
        verbose=verbose,
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
