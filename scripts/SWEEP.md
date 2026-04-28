# Sweep runner

Run the multi-turn KernelBench agent across **multiple models × levels × problems**
in parallel, with per-model rate limiting, GPU oversubscription, per-GPU
perf-timing locks, and a live self-refreshing HTML report.

## TLDR

```bash
# 1. Drop API keys in .env (TEJAS_AZURE_KEY, THAVA_AZURE_KEY, etc.).
# 2. Edit configs/sweep.example.toml to taste.
# 3. Run.
uv run python scripts/run_sweep.py config=configs/sweep.example.toml

# Live report:  http://<host>:8765/index.html
# (from your laptop:  ssh -L 8765:localhost:8765 user@host  →  http://localhost:8765)
```

## Minimal TOML

```toml
[run]
name              = "sweep_demo"
dataset_src       = "local"          # "local" | "huggingface"
variant           = "original"       # "original" | "popcorn" (local only)
levels            = [1]
problem_subset    = []               # empty = all problems in level
backend           = "cuda"
precision         = "fp32"
gpu_arch          = ["Hopper"]
timing_method     = "cuda_event"
num_correct_trials = 5
num_perf_trials    = 100
tools              = "default"

[agent]
max_turns          = 10
max_tool_calls     = 30
reasoning_effort   = "medium"

[parallelism]
num_gpu_devices    = 8
agents_per_gpu     = 3               # oversubscribe - LLM-wait dominates
perf_lock_per_gpu  = true            # serialize submit_kernel timing per GPU

[report]
enabled            = true
refresh_seconds    = 30
serve              = true
serve_host         = "0.0.0.0"
serve_port         = 8765

[[models]]
name        = "gpt-5.5"
api_kind    = "openai"               # "openai" = Responses API
base_url    = "https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"
api_key_env = "TEJAS_AZURE_KEY"
rpm         = 250
tpm         = 250000

[[models]]
name        = "DeepSeek-R1"
api_kind    = "openai_chat"          # "openai_chat" = Chat Completions
base_url    = "https://thava-openai.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
api_key_env = "THAVA_AZURE_KEY"
rpm         = 250
tpm         = 250000
```

See `configs/sweep.example.toml` for the full 5-model template.

## How parallelism is laid out

```
work matrix    = (model × level × problem_id)
process pool   = num_gpu_devices * agents_per_gpu workers
each worker    = bound to one GPU via CUDA_VISIBLE_DEVICES
per model      = mp.Manager().Semaphore  (caps in-flight LLM calls)
per GPU        = mp.Manager().Lock       (serializes submit_kernel timing)
```

Why oversubscribe? Each agent turn is dominated by LLM wait (seconds to a
minute). The GPU sits idle 80–90% of the time, so 3–4 agents per H100 keep both
budgets busy. The per-GPU lock around `submit_kernel.execute` ensures perf
timings are never collected concurrently on the same device, so wall-clock
numbers stay clean even under oversubscription.

Per-model concurrency defaults to roughly `tpm / 25_000` (≈10k tokens/turn × 2.5×
safety). Override with `max_concurrency = N` on any `[[models]]` block once you
have real token usage numbers.

## API kinds

| `api_kind`    | Endpoint shape       | Used for                                                |
|---------------|----------------------|---------------------------------------------------------|
| `openai`      | Responses API        | gpt-5.x family (supports `reasoning_effort`)            |
| `openai_chat` | Chat Completions     | DeepSeek-R1, Llama-Maverick, Kimi-K2.x, generic OpenAI  |

The two paths share tools, prompts, and trajectory format — only the wire
protocol differs. DeepSeek-R1's `reasoning_content` is captured in the
trajectory but not resent to the model on the next turn (Chat Completions
doesn't require it).

## Output layout

```
runs/{run_name}/
    sweep_config.json                     -- snapshot of the TOML at run start
    {model_name}/
        level_{L}_problem_{P}_trajectory.json
        level_{L}_problem_{P}_kernel.py   -- final submitted kernel (if any)
        level_{L}_problem_{P}_cache/      -- compile cache
    report/
        style.css
        index.html                        -- overall summary + per-model rollup
        models/{model_name}.html          -- grid of cards for one model
        t/{model}__l{L}_p{P}.html         -- TLDR + full conversation
```

The runner is **resumable**: if a trajectory JSON already exists at the
expected path, that work item is skipped and its summary is folded into the
report directly.

## Viewing the report from a laptop

The runner spawns a small `http.server` on the configured port (default 8765).

```bash
# On your laptop:
ssh -L 8765:localhost:8765 you@your-h100-box
# Then in a browser:
open http://localhost:8765/index.html
```

If `serve_host = "0.0.0.0"` and the box is reachable, you can also hit it
directly. Pages auto-refresh every 60s; the underlying HTML is regenerated
every `refresh_seconds` (30s default).

## Reading a trajectory

Each trajectory page has:

- **TLDR card** at top: outcome badge, runtime, ref runtime, speedup, and
  inline compile/correctness errors when present.
- **Per-turn sections** below: each turn shows its LLM latency, an expandable
  reasoning block (yellow border), assistant text, every `function_call` with
  truncated arguments, and every tool result. Failed tool calls are
  expanded by default; everything else is collapsed.

## Tools the agent uses

Resolved by the `tools` key in `[run]`:

- `"default"` — all tools except `profile_kernel` (which needs ncu)
- `"all"` — every tool in the registry
- `"compile_kernel,run_correctness,submit_kernel"` — explicit comma list

`submit_kernel` is always added regardless; without it the agent has no way to
finalize a run.

## Common knobs

| Where             | Knob               | Effect                                                          |
|-------------------|--------------------|-----------------------------------------------------------------|
| `[run]`           | `problem_subset`   | `[1, 2, 5]` to limit; empty list = all                          |
| `[run]`           | `variant`          | `"original"` (100 problems) vs `"popcorn"` (smaller curated set)|
| `[parallelism]`   | `agents_per_gpu`   | Higher → uses more API budget, more GPU contention              |
| `[parallelism]`   | `perf_lock_per_gpu`| Set false if you don't care about timing noise                  |
| `[[models]]`      | `max_concurrency`  | Hard cap if the heuristic is off                                |
| `[[models]]`      | `deployment_name`  | If Azure deployment name ≠ display name                         |
| `[report]`        | `refresh_seconds`  | How often to rebuild HTML; 30s is usually fine                  |

## Generating the report manually

If a sweep is already done (or you're inspecting a paused run):

```bash
uv run python scripts/build_report.py runs/sweep_demo
# → writes runs/sweep_demo/report/index.html
```
