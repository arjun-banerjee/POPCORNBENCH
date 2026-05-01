# PopcornBench — Findings & Plot Plan (gpt-5.5 sweep so far)

Snapshot of what's on disk on 2026-04-28. Numbers below merge `gpt-5.5` and
`gpt-5.5-priority` only where both ran the same problem; otherwise stick to
`gpt-5.5`.

## 0. Sweep status

| variant | tier | trajectories | usable (non-API-error) |
|---|---|---|---|
| L2-pop | single_turn        | 50 | 50 |
| L2-pop | multi_turn_default | 50 | 50 |
| L2-pop | multi_turn_all     | 14 (+18 priority dups) | 14 |
| L3-orig | single_turn       | 50 | 50 |
| L3-orig | multi_turn_default| 20 | 20 |
| L3-orig | multi_turn_all    | 50 | **0** — every trajectory is a 404 `DeploymentNotFound` LLM error |
| L3-pop | single_turn        | 13 | 13 |
| L3-pop | multi_turn_default | 5 | 5 |
| L3-pop | multi_turn_all     | 13 | **0** — same 404 errors |

**Important:** L3 `multi_turn_all` is *not* in-flight; every trajectory failed
on turn 0 with an Azure deployment 404. Treat L3-all as "not run" for any
plot. L2-pop multi_turn_all is the only complete heavy-tool tier we have.

## 1. Findings

### AGENT story (tool-tier axis on L2-popcorn — the only level with all 3 tiers)

- **Correctness rate climbs sharply with the loop, then plateaus.**
  - single_turn: 29/50 = **58%** correct
  - multi_turn_default: 43/50 = **86%** correct (+28 pp)
  - multi_turn_all: 14/14 = **100%** correct, but n is small and these are
    likely the easier resumed subset (priority duplicates: 10/18 = 56%).
  - Compile-but-wrong (the "looks right, isn't" failure):
    single 9/50, default 3/50, all 0/14. The loop's correctness checker
    almost entirely eliminates this class.

- **Median speedup *decreases* in the heavy-tool tier.**
  Across the 11 problems where both default and all produced correct kernels:
  default median **2.13×** vs all median **1.70×**. So the heavy tools don't
  unlock more performance on average — they tend to push the model toward
  safer, slower kernels.

- **Concrete underperformers (multi_turn_all worse than multi_turn_default):**
  - `1_DeepSeekMLALoRAExpansion`: 1.79× → 1.18×
  - `48_ParticleFilter`: 4.74× → 1.48× (largest regression)

  Wins go the other way for `9_MetropolisHastingsStep` (6.3× → 8.4×) and
  `41_VariationalELBO` (1.17× → 1.66×). Net: heavy tools help on a few
  Bayesian/sampling kernels, hurt on the most aggressively fused ones.

- **Tool usage is real, not ignored.** Per-trajectory averages on L2-pop:
  | tool | default | all |
  |---|---:|---:|
  | compile_kernel    | 2.4 | 3.9 |
  | run_correctness   | 3.2 | 4.5 |
  | static_check      | 1.0 | 0.2 |
  | get_gpu_specs     | 1.0 | 1.0 |
  | submit_kernel     | 0.9 | 1.0 |
  | profile_kernel    | —   | **3.4** |
  | disassemble_kernel| —   | **2.0** |
  | ert_roofline      | —   | 1.0 |

  The model genuinely calls ncu (`profile_kernel`) and `cuobjdump`
  (`disassemble_kernel`) — 5+ heavy-tool invocations per trajectory on
  average. So underperformance isn't from ignoring the tools; it's from
  acting on what the tools say (often debugging instead of optimizing).

- **L2-pop "dead zones" — problems where both single_turn AND
  multi_turn_default produce wrong kernels:**
  `5_LayerNorm_GELU_DilatedConv1d_ResAdd`,
  `11_broadcast_parameter_shard`,
  `12_BatchNorm_GELU_DilatedConv1d_ResAdd`,
  `50_HiddenMarkovForward`. Two are the same fused-conv1d block; the
  collective-comm one (`broadcast_parameter_shard`) and the HMM forward pass
  flag PopcornBench-specific gaps (collective + classical Bayesian dynamic
  programming).

### PROBLEM story (original vs popcorn on L3)

Only `single_turn` has full data on both sides (50 vs 13); `multi_turn_default`
has partial L3-orig (20) and tiny L3-pop (5); `multi_turn_all` has no usable L3
data.

- **L3 single_turn, original vs popcorn:**
  | metric | L3-original (n=50) | L3-popcorn (n=13) |
  |---|---|---|
  | compile rate | 22/50 = 44% | 6/13 = 46% |
  | correctness | 14/50 = **28%** | 5/13 = **38%** |
  | LLM-error | 8/50 | 0/13 |
  | median speedup (correct) | 1.04× | **1.90×** |

  Counterintuitive: popcorn looks slightly *easier* and faster than original
  at single_turn. The likely reason isn't that popcorn is easier — it's that
  the popcorn refs are hand-written, often un-fused PyTorch glue (Bayesian
  models, normalizing flows, EM steps) whose autograd traces are slow and
  easy to beat by hand-rolling a couple of CUDA kernels. KernelBench-original
  L3 is convnet/transformer code where torch.compile already does a lot.

- **Loop helps both, but unevenly.** L3-orig single→default: 28% → 47%
  correctness, median 1.04× → 1.36×. L3-pop single→default: 38% → 100% (5/5,
  tiny n), median 1.90× → 1.14× — speedup *drops* but correctness shoots up.
  Reading: on popcorn, single_turn lucked into fast-but-borderline kernels;
  the loop replaces them with conservative-correct ones.

- **Domain clustering of popcorn names.** L2-popcorn distribution:
  modern-arch (attention/Mamba/MoE/RoPE/MLA): 13, bio/stats (HMM, Gibbs,
  GMM, evoformer, MSA, particle filter, …): 12, collective comm: 8,
  fused-block (conv/norm/GELU): 7, "other": 9, quant/FP8: 1. L3-popcorn:
  modern-arch 7, bio/stats 5, collective 1. So PopcornBench is dominated by
  *modern-arch* and *bio/stats* — the two domains KernelBench under-covers.

## 2. Plot plan

Six plots. Each survives the partial sweep (none requires L3-multi_turn_all).
Color scheme: tier — single=`#4C72B0`, default=`#DD8452`, all=`#55A467`;
variant — original=`#7F7F7F`, popcorn=`#E94B3C`. All saved as 150 dpi PNGs in
`assets/figures/`, large fonts for slides.

| # | file | what it shows | axes | fields | contribution |
|---|---|---|---|---|---|
| 1 | `01_correctness_by_tier_l2pop.png` | Correctness, compile-but-wrong, compile-fail, error stacked per tier on L2-popcorn — the cleanest "does the loop help" picture. | x: tier (single/default/all), stacked bars summing to N. | `final_result.compiled`, `final_result.correctness`, `outcome` | **agent** |
| 2 | `02_speedup_distribution_l2pop.png` | Speedup distribution among correct kernels at each tier on L2-popcorn (boxplot + dots, log y). Highlights default's higher median than all. | x: tier; y: log10(ref_runtime/runtime); n annotated. | `final_result.runtime`, `final_result.ref_runtime`, `correctness` | **agent** |
| 3 | `03_per_problem_default_vs_all_l2pop.png` | Paired bar / arrow chart, per problem, default vs all on the 11 overlapping correct cases. Clearly shows the regression on ParticleFilter / MLA-LoRA and wins on MH / VELBO. | x: problem (sorted by Δ); y: speedup; one bar per tier. | runtime fields | **agent** |
| 4 | `04_tool_call_mix.png` | Stacked bar of average tool calls per trajectory by tier. Visualizes "the model actually uses ncu/cuobjdump/ert in the all tier." | x: tier; y: avg calls/traj; stack: tool name. | `turns[].tool_calls[].tool_name` | **agent** |
| 5 | `05_l3_original_vs_popcorn_singleturn.png` | Side-by-side bars on L3 single_turn: compile rate, correctness rate, median-speedup-of-correct, for original vs popcorn. The headline "popcorn isn't easier — popcorn refs are slower" panel. | grouped bars by metric. | same final_result fields | **problem** |
| 6 | `06_l2pop_outcome_by_domain.png` | Stacked correctness bars per popcorn domain (bio/stats, modern-arch, collective, fused-block, quant, other) — shows where Popcorn dead zones cluster. Faceted by tier (single/default), so you see which domains the loop unlocks. | x: domain; y: count; stack: correct/wrong/compile-fail/error; small multiples per tier. | name → domain map + final_result | **problem** |

Skipped on purpose:
- Anything requiring L3-multi_turn_all (would be entirely empty).
- Per-tier L3-popcorn trends (n=13/5/0, too thin for a real distribution).
- Roofline / occupancy scatters — would only have 14 points for the
  heavy-tool tier; can be added once the L2 sweep finishes resuming.

### Extended-metric plots (added per request)

Pulls from `final_result.{sol_stats, kernel_launch_stats, roofline_stats}`.
All restricted to **L2-popcorn**, correct kernels only (since uncorrect
kernels' counters are noise). Annotate n on each panel.

| # | file | what it shows | axes | fields | contribution |
|---|---|---|---|---|---|
| 7 | `07_sol_by_tier.png` | SOL score (sol_score) distribution per tier; SOL is the "% of theoretical peak" envelope. Most kernels are <5%, so plot on log-y. Median + dots. | x: tier; y: log10(sol_score). | `sol_stats.sol_score` | agent |
| 8 | `08_kernel_launches.png` | Pair: # of kernels in candidate vs ref, per tier. Scatter of (ref_num_kernels, num_kernels) with diagonal y=x; points below the line are fused better than ref. Faceted by tier. | log-log scatter. | `kernel_launch_stats.{num_kernels,ref_num_kernels,fusion_ratio}` | agent |
| 9 | `09_roofline_occupancy_dram.png` | Occupancy% vs DRAM-utilization% scatter, color-coded by tier, marker size = speedup. Shows where tools push kernels in the perf-design space. | x: occupancy%; y: dram%. | `roofline_stats.{occupancy_pct, dram_utilization_pct}` + speedup | agent |
| 10 | `10_warp_stalls.png` | Average warp-stall reason breakdown (long_scoreboard, mio_throttle, barrier, …) per tier — horizontal stacked bar. Tells you *what* the kernels are stalling on and whether tools shift the bottleneck. | y: tier; x: avg cycles, stack: stall reason. | `roofline_stats.warp_stalls.*` | agent |

### Case-study timelines (added per request)

Each is one figure showing the same problem run through all three tiers,
with a small "what happened" annotation per tier. Drawn as a horizontal
strip: tier on the y-axis, four columns (outcome | speedup | SOL | #kernels)
filled with colored cells + values. Outcome cell colored
red/yellow/green for compile_fail / wrong / correct. Tools-used row
listed under each tier as a subtitle. Strong examples picked from the
14 problems where all three tiers ran:

| # | file | story |
|---|---|---|
| 11 | `11_case_metropolis_hastings.png` | **`9_MetropolisHastingsStep`** — single_turn fails to compile at all; default fixes compile and lands 6.3×; *all* lifts to 8.4× (best in sweep for this problem). Clean "tools rescue, profiling sharpens" arc. |
| 12 | `12_case_gqa_kv_expansion.png` | **`15_GQAKVHeadExpansionAttention`** — single_turn compile_fail; default+all both crush the slow PyTorch ref by ~4700-4800× (small slow ref). Showcases that even the "default" tier is enough to unlock huge wins on un-fused refs. |
| 13 | `13_case_mla_lora.png` | **`1_DeepSeekMLALoRAExpansion`** — single_turn compiles but is *wrong* (silent correctness bug); default's `run_correctness` catches it, lands 1.79×; multi_turn_all *regresses* to 1.18× even though occupancy doubles (17%→44%). The poster child for "heavy tools occasionally pessimize." |
| 14 | `14_case_particle_filter.png` | **`48_ParticleFilter`** — single_turn wrong; default 4.74×; all 1.48× despite 340% occupancy and 22% DRAM util (vs 184% / 9% in default). Profiling pushed the model toward more-launched, higher-occupancy kernels that were actually slower. Counter-example to "more tools = faster". |

If this looks right, I'll write `scripts/make_plots.py`. Anything to drop, add,
or re-cut?
