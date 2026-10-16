# V12-VPU V4 Implementation Triage
## Deep audit of current benchmark evidence, V4 architecture proposals, and what to cut first

**Scope.** This review uses the current benchmark output in `Pasted text(20261013-185423).txt`, the preceding benchmark output in `Pasted text(20261013-183534).txt`, and the architecture ideas consolidated in `VPU_ARCHITECTURE_METHODS_ADDENDUM_v4.md`. It separates measured observations from hypotheses. It does not claim to have inspected shader source files, generated SPIR-V, the benchmark implementation, or the target GPU because those were not included in the supplied logs.

## Executive decision

Do not implement the full V4 feature list as one combined change. First fix the benchmark/reporting contract and validate shader equivalence. Then use the evidence to optimise Float32 matvec. After that, add only the V4 runtime facilities that the measured workload needs.

**Recommended order:**

1. **P0: trust the numbers** — correct report formulas and stale narrative; record device/toolchain; repeat measurements; validate that every variant performs the same work.
2. **P1: optimise large Float32 matvec** — inspect its generated shader, memory access, reduction strategy and tile layout. This is the strongest current candidate.
3. **P1: fix the quantised path's load/unpack/dot bottleneck** — do not assume the fused version wins; large direct-unfused currently beats direct-fused.
4. **P1: retain simple fusion, but make it cost-based** — fusion is a candidate optimisation, not a rule. Validate the anomalous small transform and large Q8 cases.
5. **P2: implement reusable runtime foundations** — typed IR/pass interfaces, lazy state/binding cache, transfer ring and structured telemetry.
6. **P2/P3: add queues, work elision and paging only with representative workloads** — variable-cost MoE jobs for scheduling; memory-pressure traces for paging; genuine sparse cases for work elision.
7. **Keep historic graphics features out of the LLM core** — adapt principles, do not emulate graphics blocks.

---

# 1. Benchmark evidence recalculated from the latest log

Ratios below use the best recorded time in each family for that specific shape:

`best-to-best speedup = fastest direct time / fastest V12-VPU time`

A ratio above 1.0 means V12-VPU was faster in that sample. This comparison does not by itself prove statistical significance or identify which compiler feature caused the result.

| Workload | Shape label | Best direct implementation | Best direct (µs) | Best VPU implementation | Best VPU (µs) | Ratio | VPU time change |
|---|---|---|---:|---|---:|---:|---:|
| Float32 matvec | small, 256×256 | direct_fused | 41.302 | v12_fused | 41.458 | 0.996× | 0.4% slower |
| Float32 matvec | medium, 2048×2048 | direct_tiled | 683.073 | v12_tiled | 648.750 | 1.053× | 5.0% faster |
| Float32 matvec | large, 8192×8192 | direct_tiled | 30,300.053 | v12_tiled | 21,939.947 | **1.381×** | **27.6% faster** |
| Int8 matvec | small, 256×256 | direct_fused | 42.448 | v12_fused | 44.010 | 0.964× | 3.7% slower |
| Int8 matvec | medium, 2048×2048 | direct_tiled | 300.573 | v12_tiled | 300.729 | 0.999× | 0.1% slower |
| Int8 matvec | large, 8192×8192 | direct_unfused | 4,674.427 | v12_tiled | 4,629.375 | 1.010× | 1.0% faster |
| Transform | small, 1,024 | direct_unfused | 23.542 | v12_tiled | 18.385 | 1.281× | 21.9% faster |
| Transform | medium, 65,536 | direct_fused / v12_tiled tie | 76.406 | v12_tiled | 76.406 | 1.000× | equal |
| Transform | large, 1,000,000 | direct_fused | 955.677 | v12_tiled | 965.208 | 0.990× | 1.0% slower |
| Conv2D | small, 64×64 | direct_fused | 20.677 | v12_tiled | 21.615 | 0.957× | 4.5% slower |
| Conv2D | medium, 256×256 | direct_tiled | 37.292 | v12_tiled | 37.344 | 0.999× | 0.1% slower |
| Conv2D | large, 512×512 | direct_tiled | 71.979 | v12_tiled | 75.052 | 0.959× | 4.3% slower |

These numbers are transcribed from the latest supplied report and recalculated best-to-best. The file's own `S` and prose summaries use other reference choices and contain stale/inconsistent conclusions; do not use those summaries as ground truth until repaired.

### Actual evidence from this run

- **Large F32 matvec is the clearest win.** V12 tiled is 8.36 ms faster than the best reported direct variant. The win is real as a sample comparison, but could come from shader algorithm/layout, not from the virtual ISA abstraction itself.
- **Medium F32 matvec is a small candidate win** of about 5%, requiring repeat-run spread to judge.
- **Q8 matvec is parity.** Its 1% large-shape difference is too small to justify complexity without repeated measurements.
- **Conv2D is parity-to-direct-wins** in the current sample. Do not prioritise VPU-specific convolution work from this evidence.
- **Small transform is suspicious.** `direct_unfused` (three dispatches) is 23.542 µs, while `direct_fused` (one dispatch) is 36.823 µs. V12 is faster than the slow direct fused variant, but the winning direct baseline is much closer. Confirm equivalent maths, dispatch work, data movement and timing boundaries.
- **Large Q8 direct-unfused is faster than direct-fused.** This may be a legitimate register-pressure/algorithm trade-off, or a mismatch in work/timing. It disproves the assumption that one dispatch or fusion is automatically faster.

## 2. The previous run and latest run cannot be combined

The earlier uploaded benchmark has substantially different timings for some kernels:

| Workload / variant | Earlier log (µs) | Latest log (µs) | Approximate change |
|---|---:|---:|---:|
| Conv2D small, direct_fused | 285.1 | 20.677 | 13.8× lower |
| Conv2D large, direct_fused | 378.8 | 75.573 | 5.0× lower |
| Transform small, direct_fused | 536.3 | 36.823 | 14.6× lower |
| F32 matvec large, direct_fused | 39,589.4 | 35,484.8 | 1.12× lower |
| Q8 matvec large, direct_fused | 5,919.9 | 5,777.7 | 1.02× lower |

This discrepancy might be explained by code changes, workload semantics, an earlier timing bug, pipeline setup, or different run conditions. The logs alone do not identify the cause. Preserve both runs, attach benchmark version/commit and shader hashes, and do not treat the differences as optimisation gains until the setup is reconciled.

---

# 3. Fix the benchmark before trusting optimisation decisions

## P0.1 Replace ad-hoc report arithmetic with tested metric definitions

For elapsed time in microseconds:

- `seconds = gpu_us * 1e-6`
- `GFLOP/s = floating_point_operations / (gpu_us * 1000)`
- `decimal_GB/s = bytes_transferred / (gpu_us * 1000)`
- `arithmetic_intensity = operation_count / bytes_transferred`
- `speedup = baseline_time / candidate_time`
- `time_saved_percent = 100 * (baseline_time - candidate_time) / baseline_time`

Do not calculate `GFLOP/s`, `GB/s`, or “compute-bound/memory-bound” from unknown operation counts or guessed byte traffic. Define whether one multiply-add counts as one MAC or two FLOPs. Record exact tensor shapes, data types, strides, padding, scales, and bytes read/written for each kernel.

### Sanity check for the large F32 matvec

If the operation really reads one dense 8192×8192 FP32 matrix once:

- Matrix payload alone: `8192 * 8192 * 4 = 268,435,456 bytes` (256 MiB).
- Approximate arithmetic using two FLOPs per multiply-accumulate: `2 * 8192 * 8192 = 134,217,728 FLOPs`.
- At 21,939.947 µs, that implies approximately **12.2 decimal GB/s** and **6.12 GFLOP/s**, excluding small vector/output traffic.

The latest report prints `GBW/s` around `0.001` and `GFLOPS` around `0.001` for this case, which does not match these assumptions. Either the report formula is wrong or the kernel is not doing the expected full dense operation/reading the expected bytes. Resolve this before using the report's roofline classification.

For an 8192×8192 byte-per-weight Q8 matrix, the weight payload is about 64 MiB before row scales and other metadata. At 4.629 ms, the payload-only estimate is about 14.5 decimal GB/s. Recalculate with the true packed format and observed workload.

## P0.2 Validate apples-to-apples semantics

For every shader variant, save a machine-readable workload contract:
- operation graph and operation count
- M/N/K, strides and layout
- input/output type and quantisation scale/zero-point
- accumulation precision, saturation/overflow behaviour and tolerances
- dispatch count and workgroup dimensions
- bytes read/written and expected outputs
- source and SPIR-V hashes

Run multiple random seeds, awkward non-multiple tile shapes, tiny dimensions, tails, zeros, positive/negative values, and appropriate quantisation edge cases. Use a CPU reference plus cross-variant comparison. For floating point, report max absolute error, max relative error and RMS error, rather than only `correct=True`.

## P0.3 Make timing stable and separately report the timing boundaries

The logs do not include the GPU identity, driver, Vulkan version, clock/power state, timing query configuration, sample count or warm-up count. Add those fields to every result.

Recommended protocol:
1. Create buffers, descriptor sets, pipelines and pipeline cache before steady-state measurement.
2. Validate the selected queue supports timestamps and convert query ticks using `timestampPeriod`; handle the queue's `timestampValidBits`.
3. Warm each pipeline, then run repeated batches. A practical start is 10–20 warm-up iterations and at least 30 timed batches; each batch should repeat enough dispatches to produce a stable interval.
4. Randomise or alternate variant order to minimise clock/thermal/order bias.
5. Report median, p10/p90 or median absolute deviation, min, and sample count. Keep compilation/cold-start results separate.
6. Report three timings separately: GPU execution, host submission/recording, and end-to-end (including required copies/waits).
7. Do not perform host readback or `vkQueueWaitIdle` per dispatch within a steady-state timing loop. Synchronise at the batch boundary.
8. Add correctness and metric-formula unit tests to CI.

Vulkan timestamp queries are the right primitive for device timing, but valid timestamp bits and `timestampPeriod` have to be respected. The Vulkan profiling tutorial recommends warming the workload before timing and using roofline arithmetic intensity only with defensible FLOP/byte counts.

References:
- https://docs.vulkan.org/spec/latest/chapters/queries.html
- https://docs.vulkan.org/guide/latest/profiling.html
- https://docs.vulkan.org/tutorial/latest/ML_Inference/Debugging_and_Performance/04_performance_profiling.html

## P0.4 Regenerate all narrative from the current result data

The current log's raw rows say Conv2D medium is about parity and Conv2D VPU is slower on small/large; the later prose still claims a 1.461× medium win from older results. The analysis section says small transform is 1.94× faster relative to `direct_fused`, while the best direct implementation is `direct_unfused` and the best-to-best ratio is about 1.28×. This is report-generation drift.

Remove hard-coded narrative and stale speedups. Generate every statement/table from the same JSON file, and include:
- same-algorithm comparison
- best direct vs best VPU comparison
- time delta in percent
- statistical interval/spread
- explicit label identifying the baseline
- “inconclusive” where variance overlaps the difference

---

# 4. Kernel-level optimisation plan informed by V4

## P1. Float32 matrix-vector multiplication: first performance target

**Current evidence:** V12 tiled is the best reported large variant at 21.94 ms versus direct tiled at 30.30 ms. Medium V12 tiled is about 5% faster; small is parity. The unfused stage report says the first stage dominates, but that does not reveal the internal behaviour of the winning fused/tiled kernels.

### Investigate in this order

1. **Inspect the exact generated GLSL and SPIR-V for direct_tiled vs v12_tiled.** Compare thread-to-output mapping, vectorisation, loop unrolling, reductions, bounds checks, layout, workgroup dimensions and compiler specialisations. Save hashes and disassembly. The VPU's advantage needs an identified mechanism, not just a faster label.
2. **Validate the true input traffic.** Confirm all 8192×8192 elements are read and every output row is written exactly once. Use device counters/vendor tooling if available.
3. **Autotune tile shapes**, but distinguish *M/output-row tiling* from *K/reduction tiling*. Sweep a modest grid such as output rows per workgroup, K chunk size, workgroup size and vector load width; reject configs that spill or reduce occupancy.
4. **Compare reduction topologies:** one invocation per output row, subgroup cooperation over K, several rows per subgroup/workgroup, and workgroup-shared partial reduction. Query actual subgroup capabilities/size and do not assume warp width.
5. **Coalesce matrix reads.** Consecutive lanes should access adjacent weights; check alignment, stride and tails. Reuse input-vector values only when the cache/dispatch evidence shows a benefit.
6. **Track register pressure and occupancy** for each shape. More fusion/unrolling may decrease global traffic but increase registers, reduce active waves or cause spills.
7. **Compare against a genuinely tuned direct kernel** using the same tiling search budget. Otherwise the test proves only that one kernel is better than another, not that the VPU abstraction adds value.

Useful modern Vulkan building blocks include compute workgroups, shared memory and subgroup operations. Prefer subgroup-local communication and reductions when applicable, using shared memory only when cross-subgroup communication or reuse makes it worthwhile:
- https://docs.vulkan.org/guide/latest/compute_shaders.html
- https://docs.vulkan.org/tutorial/latest/Advanced_Vulkan_Compute/04_Subgroup_Operations/02_cross_invocation_communication.html

## P1. Int8/Q8 matvec: focus on the dominant load/decode/dot path

**Current evidence:** the large run is around parity. The fastest direct result is `direct_unfused` at 4.674 ms, while `v12_tiled` is 4.629 ms. At small size, direct fused is faster. No robust VPU advantage is demonstrated.

### Optimisation experiments

1. Inspect whether bytes are loaded efficiently, how packed int8 values are unpacked/sign-extended, and whether quantisation scales are reused or reloaded.
2. Determine if `int8_dot` is genuinely the dominant stage or the report's stage label is merely inferred. Tie each label to the exact dispatch/kernel name.
3. Compare packed word/vector loads and multiple dot-product formulations. Only rely on hardware acceleration if the actual Vulkan device and compiled shader support it.
4. Measure fused and split variants. Do not force fusion: `direct_unfused` already beats `direct_fused` on the large sample, and a fused shader may have higher register pressure or poor instruction scheduling.
5. Separate weight-byte throughput, scale metadata, unpack/dequant cost and accumulator cost in profiler analysis.
6. Include the actual model quantisation format later. A simple signed-int8 microbenchmark is not evidence about GGUF Q4/Q5/Q8 block layouts.

## P1. Transform chain: fix the suspicious small result, then keep only the best path

**Current evidence:** VPU's apparent small win depends on a slower `direct_fused` result. The direct unfused three-dispatch variant is already 23.54 µs versus 36.82 µs for direct fused, while V12 tiled is 18.39 µs. Large transform is effectively parity or slightly worse in VPU.

Actions:
- Verify all variants calculate identical transform → scale/bias → dot-product semantics, element counts and buffer traffic.
- Check whether the direct-fused compiler path creates excessive live values/registers or contains a bounds/layout inefficiency.
- Benchmark single-op, two-op and full fused sequences; use a fusion cost model rather than assuming one dispatch wins.
- Optimise the small case for launch/dispatch cost and the large case for memory layout/throughput. They are different regimes.
- Keep V12 transform as a candidate only if the speedup survives repeated runs and equally tuned direct shaders.

## P2. Conv2D: do not prioritise VPU-specific work yet

Current best-to-best results favour direct Vulkan by about 0–4.5%, which may be noise but do not show a VPU win. Keep the fastest direct kernel as baseline. Only revisit VPU-specific convolution if:
- repeated runs demonstrate a stable gap;
- the VPU compiler creates a materially different tile/fusion/data-reuse strategy;
- the workload matches actual inference kernels or vision preprocessing.

Do not transfer Kyro's 32×16 screen tile as a fixed compute tile. Kyro's documented tile-based deferred rendering avoids shading hidden pixels and keeps per-tile render data local, which is a graphics-specific visibility pipeline. Transfer locality and reuse principles, and tune tile sizes for the actual compute device.

PowerVR reference: https://docs.imgtec.com/starter-guides/powervr-architecture/html/topics/tile-based-deferred-rendering-index.html

---

# 5. V4 feature-by-feature disposition

| V4 feature | Disposition now | Reason / first test |
|---|---|---|
| Benchmark trust, device metadata, metric correctness | **Implement first (P0)** | All later optimisation decisions depend on this. |
| Typed backend-independent IR and pass interface | **Keep; implement minimal version after P0** | Useful architecture boundary, but initial pass set should stay small. |
| Fusion cost model | **Implement early** | Evidence shows large penalty for unfused transform/F32 paths, but Q8 shows fusion is not always fastest. Model code size/register pressure as well as launches and memory writes. |
| Tile planner + autotuning | **Implement early for F32 matvec** | Current strongest win; compare same candidate sweep on direct and VPU paths. Do not fix tile dimensions globally. |
| Precision policy | **Keep, but initially use only tested formats** | Make storage, compute, accumulation and output types explicit. Test Q8 and FP32 correctness before adding more precision modes. |
| SIMD/subgroup-aware layout | **High priority for matvec after device inspection** | Test actual subgroup support and register/shared-memory trade-offs. |
| Lazy state / pipeline / descriptor cache | **Keep but measure CPU-side first** | Can reduce host recording/binding overhead, but GPU timestamp alone will not show that benefit. Benchmark many tiny dispatches and full end-to-end runtime. |
| Transfer ring / reusable staging buffers | **Implement foundationally if copies/allocation are visible** | Measure host upload, copies, allocation count and wait time. Do not emulate DirectBurst as a separate physical memory class. |
| Tensor/weight residency manager | **Design now; implement with memory-pressure harness** | Valuable for large models/MoE, but current resident microbenchmarks do not prove it helps. Track budget/usage and paging bytes. |
| Hot/warm/cold paging | **High for the eventual low-VRAM goal; separate project milestone** | Test a model/working set that exceeds the available device budget. Include throughput and tail latency, not just capacity. |
| CPU-managed dynamic work bins | **Prototype after stable kernels** | The Stanford queue/bin model fits variable-cost MoE tasks. Test with a skewed expert/request trace; fixed matrix microbenchmarks cannot prove value. |
| Persistent GPU task queue | **Defer** | More difficult correctness, occupancy, termination and atomic-contention risks; prove CPU-managed queue overhead first. |
| Device-generated commands | **Defer / capability-gated** | Not portable baseline behaviour; indirect dispatch is not the same as arbitrary device-side command creation. |
| Work elision | **Keep as an optional pass, implement only with real sparse workloads** | Skip work only when masks, routing or exact data semantics prove that outputs cannot change. Graphics occlusion is not generic AI sparsity. |
| Deadline scheduler / decode-vs-prefill priority | **Defer until server workloads exist** | Requires TTFT, inter-token latency, throughput and fairness measurements, not isolated shader times. |
| Cobalt UMA/large shared pool inspiration | **Keep as memory-backend abstraction only** | Device-local VRAM and host-visible/shared RAM have different bandwidth/latency. Use Vulkan memory-budget data where available; do not imply 64 GB of system memory behaves like 64 GB of VRAM. |
| Kyro tile/deferred rendering | **Keep locality and work-binning principles; strike literal graphics emulation** | Visibility rejection and screen tiles do not map directly to dense LLM arithmetic. |
| Wildcat dual-VPU/crossbar simulation | **Strike literal emulation; keep cost-balanced partitioning** | The host GPU already has its own workgroup scheduler. Software partitioning must show better load balance to justify itself. |
| DirectBurst | **Strike as a literal hardware model; keep transfer-ring/batched upload concept** | Reusable buffers and reduced copies are the relevant modern abstraction. |
| 64-bit accumulation as a speed optimisation | **Strike from default path; optional correctness mode only** | It may be slower and is not supported uniformly. |
| SGI rasteriser, Z/stencil/alpha/chroma, overlay, stippling, frame lock, exact 36-bit stage precision | **Strike from LLM compute core** | Retain only in an optional graphics/vision compatibility module if that becomes an explicit product requirement. |
| Texture anisotropy/filtering used as general AI/embedding primitives | **Strike from LLM core** | Use native buffer/vector/matrix kernels; interpolation stays as an optional vision/data-resampling primitive. |
| A monolithic `vpu_combined` as default | **Defer** | Keep all passes independently switchable and use ablations to let per-workload tuning choose the combination. |

---

# 6. The implementation order I would actually follow

## Milestone 0: Repair the benchmark (first change set)
- Fix all unit conversions and FLOP/byte definitions.
- Generate summary prose only from the current JSON.
- Add JSON schema and report self-tests with known inputs/expected metric outputs.
- Add device/driver/Vulkan/compiler/feature/hash metadata.
- Add repeated timing batches, warm-up, median and variability.
- Validate semantic equivalence and write a workload contract for every shader.

**Exit condition:** rerunning the same unchanged build on the same machine produces stable results within a documented variance band, and metric unit tests pass.

## Milestone 1: F32 matvec (first kernel optimisation)
- Audit shader and SPIR-V for `direct_tiled` and `v12_tiled`.
- Verify full 8192×8192 work/bytes.
- Build a small parameter sweep: rows/output per workgroup, reduction chunk, local size, vector width and subgroup reduction.
- Profile registers, occupancy and global load behaviour.
- Tune the direct baseline with the same sweep.
- Re-run small/medium/large shapes, including irregular dimensions.

**Exit condition:** VPU's best-to-best improvement is repeatable, its uncertainty band excludes parity, and the report explains which lower-level change produced it.

## Milestone 2: Q8 path and fusion economics
- Validate packed format/scales and actual int8 arithmetic.
- Fix kernel-stage naming and per-stage timing.
- Compare direct/VPU fused and split variants.
- Test one/two/four/eight elements per invocation only where alignment and shader semantics permit.
- Track register usage and bytes per matrix value.

**Exit condition:** choose the correct variant per shape; do not force fusion where direct-unfused remains faster.

## Milestone 3: virtual compiler foundations
- Typed IR; instruction validation; aliasing/dependency model.
- Composable passes: canonicalise → fuse (cost model) → tile/layout → precision lower → generate shader.
- Cache generated SPIR-V and pipelines by all shape/type/device feature keys.
- Keep the native baseline independently authored.

**Exit condition:** pass ablations show which passes help, and disabled passes do not change semantics.

## Milestone 4: V4 runtime features that address real engine problems
- Transfer ring and explicit transfer/computation dependencies.
- Lazy state/bindings, measuring CPU command-recording time.
- CPU-managed work bins and cost-aware scheduling using realistic MoE traces.
- Tensor/weight residency manager tested under controlled memory pressure.
- Optional work elision on genuine sparse/inactive work.
- Deadline-aware decode/prefill scheduling only after server-level metrics are in place.

**Exit condition:** improvements are seen in end-to-end inference (tokens/s, time-to-first-token, inter-token latency, peak VRAM, paging stalls) and not merely in an isolated shader.

## Milestone 5: only then evaluate advanced queueing
- Persistent GPU task queue.
- Device-generated commands, only if the Vulkan device exposes the required extension and the use case benefits.
- Hybrid CPU/GPU work distribution.
These are optional research paths, not required foundations.

---

# 7. First items to strike or rewrite in the previous V4 material

1. **Delete the stale benchmark narrative.** The claim of a 1.461× Conv2D medium win belongs to a different run and conflicts with current rows. The current best-to-best result is effectively parity.
2. **Replace mean speedup summaries** that mix kernels, sizes or reference variants. Report every shape and baseline; if an aggregate is needed, use a clearly stated aggregation of per-shape ratios and still publish the rows.
3. **Strike the claim that V12 itself is the cause of speedups.** The current data only compare kernel variants. Until direct and VPU codegen use equivalent optimisation opportunities and the source/SPIR-V difference is understood, credit the winning shader strategy, not the virtual architecture.
4. **Remove literal historical hardware emulation from the AI core.** No Cobalt ASIC emulation, Kyro visibility pipeline, Wildcat crossbar, graphics rasterisation, or historical pipeline bit widths in the main inference path.
5. **Remove fixed tile dimensions as architecture rules.** Kyro's 32×16 pixel tile is a historical screen-space tile, not a universal Vulkan compute optimum.
6. **Remove “fusion always wins”.** Make fusion cost-based and retain split variants; Q8 currently demonstrates a case where direct-unfused is best on the large shape.
7. **Defer persistent GPU queues and device-generated commands** until CPU-managed queues show a measurable problem they can solve.
8. **Move graphics functions to an optional vision/compatibility module.** Cobalt YCrCb/RGB conversion and 4×4 colour transforms can be useful for vision/video preprocessing, but depth/stencil/overlays/alpha/chroma/frame-lock are not core LLM instructions.
9. **Remove “64 GB memory” as a synonym for 64 GB of fast VRAM.** Implement a memory backend/residency policy and report heap budget, usage, transfer cost and page faults/evictions.
10. **Consolidate V4 documentation.** The current file named `..._v4.md` begins with a v2 title and appends v3/v4 sections. Before implementation, make one canonical V4 spec with a current title, one prioritisation matrix and an explicit deferred/rejected list. Keep source notes/history in an appendix.

---

# 8. What the earlier architecture papers contribute—and what they do not prove

- **PowerVR Kyro/Kyro II:** transfer tile locality and early rejection where work is semantically unnecessary. The original hidden-surface removal and on-chip pixel buffering are graphics mechanisms, not general LLM instructions. PowerVR's own documentation explains its tile-based deferred rendering model: https://docs.imgtec.com/starter-guides/powervr-architecture/html/topics/tile-based-deferred-rendering-index.html
- **Chromium / Greg Humphreys:** composable stream transformations, lazy state updates, bucketing, and explicit trade-offs between sorting and load balance. This supports the V4 pass pipeline and state cache, but does not prove these approaches are faster in this Vulkan workload. Thesis: https://graphics.stanford.edu/papers/humper_thesis/humper_thesis.pdf
- **Stanford GPU notes:** work queues/bins can collect coherent work asynchronously and offer a more flexible alternative to fixed batching, if there is enough intrinsic workload coherence. Test dynamic bins on skewed MoE/request workloads, not only regular matvecs: https://graphics.stanford.edu/~yoel/notes/index-0507.html
- **Cobalt / Wildcat Realizm:** memory pooling, transfer organisation, load balancing, precision decisions and accumulation are architectural inspiration. The relevant modern adaptation is buffer/residency/scheduling policy, not a cycle-accurate reproduction.
- **Vulkan:** shared memory, subgroups, indirect dispatch, explicit synchronisation and memory-budget reporting are modern primitives to evaluate. They do not guarantee a speedup without device-specific profiling.

Relevant Vulkan documentation:
- Timestamp queries and timestamp period: https://docs.vulkan.org/spec/latest/chapters/queries.html
- Vulkan profiling: https://docs.vulkan.org/guide/latest/profiling.html
- Compute shaders, workgroups and indirect dispatch: https://docs.vulkan.org/guide/latest/compute_shaders.html
- Vulkan memory heaps and `VK_EXT_memory_budget`: https://docs.vulkan.org/spec/latest/chapters/memory.html
- Subgroup communication and reductions: https://docs.vulkan.org/tutorial/latest/Advanced_Vulkan_Compute/04_Subgroup_Operations/02_cross_invocation_communication.html

---

# Final recommendation

**Before V4 feature implementation, strike scope—not the useful ideas.** Keep the virtual IR, native Vulkan baseline, cost-based fusion, tile/autotune, precision policy and memory/scheduling abstractions. Remove literal historical graphics emulation from the inference core. Defer advanced GPU queues and full paging until their motivating workloads and measurements exist.

The very next code change should be a benchmark-integrity patch. The very next performance change after that should be an isolated F32 matvec shader audit and tile/reduction sweep. That sequence prevents attractive architectural ideas from being layered on top of incorrect metrics or a weak baseline.
