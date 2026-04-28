# Sweep runner

Run the multi-turn KernelBench agent across **multiple models × levels × problems**
in parallel, with per-model rate limiting, GPU oversubscription, per-GPU
perf-timing locks, and a live self-refreshing HTML report.

## TLDR

```bash
# 1. Drop API keys in .env (TEJAS_AZURE_KEY, THAVA_AZURE_KEY, etc.).
# 2. Edit configs/sweep.example.toml to taste.
# 3. Run.
uv run python scripts/run_sweep.py configs/sweep.example.toml

# Live report:  http://<host>:8765/index.html
# (from your laptop:  ssh -L 8765:localhost:8765 user@host  →  http://localhost:8765)
```

## Quick correctness test (a few problems on level 1)

Pin `problem_subset` to a short list and drop everything else to a single
model with shallow agent caps. This is the fastest way to check that an
endpoint, key, and the eval loop all wire up before you commit a real run.

```toml
# configs/sweep.smoke.toml
[run]
name              = "smoke_l1"
dataset_src       = "local"
variants          = ["original"]
levels            = [1]
problem_subset    = [1, 19, 23]      # Square_matmul, ReLU, Softmax
backend           = "cuda"
precision         = "fp32"
gpu_arch          = ["Hopper"]
timing_method     = "cuda_event"
num_correct_trials = 3               # quicker than the default 5
num_perf_trials   = 20               # quicker than the default 100
tools             = "compile_kernel,run_correctness,submit_kernel"

[agent]
max_turns          = 4
max_tool_calls     = 8
reasoning_effort   = "low"

[parallelism]
num_gpu_devices    = 1
agents_per_gpu     = 2
perf_lock_per_gpu  = true

[report]
enabled            = true
refresh_seconds    = 15
serve              = true
serve_port         = 8765

[[models]]
name        = "gpt-5.5"
api_kind    = "openai"
base_url    = "https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1/"
api_key_env = "TEJAS_AZURE_KEY"
rpm         = 250
tpm         = 250000
```

```bash
uv run python scripts/run_sweep.py configs/sweep.smoke.toml
```

After it finishes, the top of `report/index.html` shows the correct/compiled
counts; click into the variant card → into the model card → into any single
problem to see the conversation, the kernel that was submitted, and the
correctness verdict.

## Minimal TOML

```toml
[run]
name              = "sweep_demo"
dataset_src       = "local"          # "local" | "huggingface"
variants          = ["original", "popcorn"]   # sweep axis (or single string `variant = "..."`)
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
base_url    = "https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1/"
api_key_env = "TEJAS_AZURE_KEY"
rpm         = 250
tpm         = 250000

[[models]]
name        = "DeepSeek-R1"
api_kind    = "openai_chat"          # "openai_chat" = Chat Completions
base_url    = "https://thava-openai.services.ai.azure.com/models"
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

## Endpoints (Azure-specific)

The five models we sweep sit on two different Azure surfaces. The base URLs
were verified end-to-end by `scripts/probe_endpoints.py` (one trivial request
per model) — re-run that script if you suspect an outage or rotated key.

### Azure OpenAI v1 preview (`...cognitiveservices.azure.com`)

The OpenAI-branded models (gpt-5.x) live here. The OpenAI Python SDK auto-
appends `/responses` or `/chat/completions` to the v1 base, so one URL covers
both `api_kind`s. **No `api-version` query string required.**

```toml
base_url = "https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1/"
api_kind = "openai"          # gpt-5.x supports the Responses API
```

### Azure AI Inference / Foundry (`...services.ai.azure.com`)

DeepSeek-R1, Llama-Maverick, and Kimi-K2.6 are deployed on Azure AI Inference
(not Azure OpenAI). The SDK base is `/models`; only Chat Completions is
supported here, never the Responses API.

```toml
base_url = "https://thava-openai.services.ai.azure.com/models"
api_kind = "openai_chat"
```

### Verified working today

| Model                                    | base_url                            | api_kind     |
|------------------------------------------|-------------------------------------|--------------|
| gpt-5.4-pro                              | `…cognitiveservices…/openai/v1/`    | `openai`     |
| gpt-5.5                                  | `…cognitiveservices…/openai/v1/`    | `openai`     |
| DeepSeek-R1                              | `…services.ai.azure.com/models`     | `openai_chat`|
| Llama-4-Maverick-17B-128E-Instruct-FP8   | `…services.ai.azure.com/models`     | `openai_chat`|
| Kimi-K2.6                                | `…services.ai.azure.com/models`     | `openai_chat`|

### Quirks worth knowing

- **DeepSeek-R1 reasoning is inline.** It emits `<think>...</think>` blocks
  *inside the assistant `content`* rather than via the `reasoning_content`
  field that our agent reads. So the agent will treat the reasoning as part of
  the assistant's normal text. The trajectory still captures it; the HTML
  report just won't show it under the dedicated "reasoning" expander for that
  model.
- **`reasoning_effort` is silently ignored** when `api_kind = "openai_chat"`.
  Chat Completions has no `reasoning.effort` field. Set it on the gpt-5.x
  entries only.
- **Rotate any key that lands in the working tree.** `tmp.py` and similar
  scratch files are not committed but they're on disk; treat any key inside
  them as compromised.

## Output layout

```
runs/{run_name}/
    sweep_config.json                            -- snapshot of the TOML at run start
    {variant}/{model_name}/
        level_{L}_problem_{P}_trajectory.json
        level_{L}_problem_{P}_kernel.py          -- final submitted kernel (if any)
        level_{L}_problem_{P}_cache/             -- compile cache
    report/
        style.css
        index.html                               -- all-variants overview (one card per variant)
        v/{variant}/index.html                   -- per-variant summary + per-model rollup
        v/{variant}/models/{model}.html          -- grid of cards for one (variant, model)
        v/{variant}/t/{model}__l{L}_p{P}.html    -- TLDR + full conversation
```

When you sweep multiple variants (e.g. `variants = ["original", "popcorn"]`),
the top-level `index.html` shows one card per variant with rollup stats; click
a card to drop into that variant's subreport. This makes the original-vs-popcorn
comparison the first thing you see.

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

## TOML reference

Every option, grouped by section. Required fields are marked **req**; the rest
have sensible defaults.

### `[run]` — what to evaluate

| Key                  | Type             | Default                                | Notes |
|----------------------|------------------|----------------------------------------|-------|
| `name`               | string           | **req**                                | Used as the directory name under `runs/`. |
| `runs_dir`           | string           | `"runs"`                               | Where to write the run directory. Relative paths are resolved from the repo root. |
| `dataset_src`        | string           | **req**                                | `"local"` (reads `KernelBench/level{N}/{variant}/`) or `"huggingface"`. |
| `dataset_name`       | string           | `"ScalingIntelligence/KernelBench"`    | Only used when `dataset_src = "huggingface"`. |
| `variants`           | list of strings  | `["original"]`                         | Sweep axis. Each entry is a subdirectory under `KernelBench/level{N}/`. Common values: `"original"` (100 upstream problems) and `"popcorn"` (curated set). Local only. |
| `variant`            | string           | —                                      | Legacy single-variant form. Equivalent to `variants = [variant]`. |
| `levels`             | list of ints     | **req**                                | E.g. `[1]`, `[1, 2]`, or `[1, 2, 3, 4]`. |
| `problem_subset`     | list of ints     | `[]`                                   | **Empty list runs every problem in the level.** Pass IDs to limit, e.g. `[1, 19, 23]`. The same subset filter is applied to every (level, variant) combination — IDs that don't exist in a given variant are silently dropped. |
| `backend`            | string           | `"cuda"`                               | `"cuda"` \| `"triton"` \| `"hip"` \| `"tilelang"` \| `"thunderkittens"` \| `"cute"`. |
| `precision`          | string           | `"fp32"`                               | `"fp32"` \| `"fp16"` \| `"bf16"`. Auto-overridden to `fp16` for tilelang and `bf16` for thunderkittens. |
| `gpu_arch`           | list of strings  | `["Hopper"]`                           | Passed to `set_gpu_arch()`. For H100 use `["Hopper"]`; for A100 `["Ampere"]`. |
| `timing_method`      | string           | `"cuda_event"`                         | `"cuda_event"` \| `"do_bench"` \| `"host_time"`. |
| `num_correct_trials` | int              | `5`                                    | Correctness trials run by `run_correctness` and inside `submit_kernel`. |
| `num_perf_trials`    | int              | `100`                                  | Timing trials inside `submit_kernel`. Drop to ~20 for smoke tests. |
| `tools`              | string           | `"default"`                            | `"default"` (everything except `profile_kernel`), `"all"`, or a comma list like `"compile_kernel,run_correctness,submit_kernel"`. `submit_kernel` is always included regardless. |

#### How to specify problems

```toml
problem_subset = []                 # all problems in each level
problem_subset = [1, 19, 23]        # exactly these IDs
problem_subset = [1, 2, 3, 4, 5]    # IDs 1-5
```

The matrix that gets executed is `models × variants × levels × problem_subset`,
so `levels = [1, 2]` plus `problem_subset = [1, 19]` runs problems 1 and 19 on
both levels (4 problems × n_models). To target *only* a specific level, use a
single-element `levels` list and put your IDs in `problem_subset`.

Problem IDs come from the filename prefix. To find an ID, look at the file
listing under `KernelBench/level1/original/`:

```bash
ls KernelBench/level1/original | head -5
# 1_Square_matrix_multiplication_.py     -> ID 1
# 10_3D_tensor_matrix_multiplication.py  -> ID 10
# 19_ReLU.py                             -> ID 19
```

### `[agent]` — agent loop caps

| Key                    | Type   | Default | Notes |
|------------------------|--------|---------|-------|
| `max_turns`            | int    | `10`    | Hard cap on LLM turns per problem. Each turn = one `responses.create` (or `chat.completions.create`) call plus its tool dispatches. |
| `max_tool_calls`       | int    | `30`    | Hard cap on total tool calls across all turns of one problem. |
| `warn_turns_remaining` | int    | `2`     | Inject a "you are running out of turns" notice this many turns before exhaustion. |
| `turn_delay_s`         | float  | `0.0`   | Sleep between turns. Usually leave at 0; the per-model semaphore already paces. |
| `reasoning_effort`     | string | unset   | `"minimal"` \| `"low"` \| `"medium"` \| `"high"`. Sent as `reasoning.effort` on the Responses API. Ignored for `api_kind = "openai_chat"`. Models can override per-entry (see `[[models]]`). |

### `[parallelism]` — process pool layout

| Key                 | Type | Default | Notes |
|---------------------|------|---------|-------|
| `num_gpu_devices`   | int  | `1`     | Real CUDA devices available. Each worker is pinned to one via `CUDA_VISIBLE_DEVICES`. |
| `agents_per_gpu`    | int  | `1`     | Oversubscription factor. Total worker count = `num_gpu_devices × agents_per_gpu`. Set to 3–4 on H100s — agent turns are LLM-bound, so each GPU sits idle most of the time. |
| `perf_lock_per_gpu` | bool | `true`  | If true, every `submit_kernel` call on a given device serializes via a per-GPU `mp.Lock`, so wall-clock timing is never measured concurrently with another agent's compile. Set to `false` only if you don't care about timing noise. |

### `[report]` — HTML report + live server

| Key               | Type   | Default     | Notes |
|-------------------|--------|-------------|-------|
| `enabled`         | bool   | `true`      | If false, no HTML is generated at all. |
| `refresh_seconds` | int    | `30`        | Background thread regenerates the report this often while a sweep runs. |
| `serve`           | bool   | `true`      | If true, also launch a small `http.server` thread serving `runs/{name}/report/`. |
| `serve_host`      | string | `"0.0.0.0"` | Bind address. Use `"127.0.0.1"` to keep it strictly local + SSH-forwarded. |
| `serve_port`      | int    | `8765`      | TCP port. Forward from your laptop with `ssh -L 8765:localhost:8765 user@host`. |

### `[[models]]` — one entry per model to sweep

This is a TOML array of tables — repeat the `[[models]]` block for each model
you want to run.

| Key                | Type   | Default                                                      | Notes |
|--------------------|--------|--------------------------------------------------------------|-------|
| `name`             | string | **req**                                                      | Display name. Also used as the directory under `runs/{name}/{variant}/{name}/`. |
| `api_kind`         | string | `"openai"`                                                   | `"openai"` for the Responses API (gpt-5.x), `"openai_chat"` for Chat Completions (DeepSeek-R1, Llama-Maverick, Kimi). |
| `base_url`         | string | **req**                                                      | The full endpoint URL; for Azure include `?api-version=...`. |
| `api_key_env`      | string | **req**                                                      | Name of the env var holding the API key. The runner reads `os.environ[api_key_env]`. |
| `deployment_name`  | string | falls back to `name`                                         | If your Azure deployment name differs from the display name, put the deployment name here. |
| `reasoning_effort` | string | inherits `[agent].reasoning_effort`                          | Per-model override. Only meaningful for `api_kind = "openai"`. |
| `rpm`              | int    | `250`                                                        | Requests-per-minute budget for this model. Currently informational; concurrency is capped via `tpm` (see below). |
| `tpm`              | int    | `250000`                                                     | Tokens-per-minute budget. Default per-model concurrency is `tpm / 25_000` (≈10k tokens/turn × 2.5× safety). |
| `max_concurrency`  | int    | derived from `tpm`                                           | Hard cap on in-flight LLM calls for this model. Override once you've measured real token usage per turn. |

### Putting it together: sweep matrix

```
work_items = (variants × levels × problem_subset × models)
process pool size = parallelism.num_gpu_devices × parallelism.agents_per_gpu
per-model concurrency cap = max_concurrency or floor(tpm / 25_000)
per-GPU perf-timing lock   = parallelism.perf_lock_per_gpu
```

Each work item lives in its own JSON trajectory file under
`runs/{name}/{variant}/{model}/level_{L}_problem_{P}_trajectory.json`. If that
file already exists when the runner starts, the work item is skipped — so
re-running the same TOML resumes a partial sweep.

## Probing endpoints before a real sweep

Before a long run, sanity-check that every key + URL combination still works:

```bash
uv run python scripts/probe_endpoints.py
# OK    gpt-5.4-pro                             openai       pong
# OK    gpt-5.5                                 openai       pong
# OK    DeepSeek-R1                             openai_chat  <think>...</think>
# OK    Llama-4-Maverick-17B-128E-Instruct-FP8  openai_chat  Pong!
# OK    Kimi-K2.6                               openai_chat  <no text>
# 5/5 endpoints reachable
```

Each probe sends a one-token "ping" through the same OpenAI SDK code path the
sweep uses, so a 200 here means the sweep can talk to that model.

## Generating the report manually

If a sweep is already done (or you're inspecting a paused run):

```bash
uv run python scripts/build_report.py runs/sweep_demo
# → writes runs/sweep_demo/report/index.html
```
