# V12GPU-Experiments

# VPU Architecture Methods Addendum v2
## SGI VPro V12 + PowerVR Kyro/Kyro II + 3Dlabs Wildcat Realizm

This addendum turns documented historical graphics techniques into testable ideas for a Vulkan-first AI compute runtime. None of the historical GPUs is assumed to have been an AI processor. Each adaptation is a hypothesis requiring an ablation benchmark.

## 1. Source-specific Wildcat Realizm findings

The supplied 3Dlabs whitepaper, *New Wildcat Realizm Graphics Technology: Bigger Bandwidth, Greater Programmability, Higher Precision* (© 2004), describes the following:

- A **programmable, compiler-centric pipeline** with programmable vertex, fragment and pixel stages. It argues against relying exclusively on fixed-function operations when algorithms evolve.
- In the VPU pipeline, a vertex stage with **16-way vertex processing** and up to **1K real machine-code instructions**, a rasterisation stage capable of **64 pixels per clock**, and a fragment stage with **48 independent processors** and up to **256K real machine-code instructions**.
- **Hierarchical Z-buffer culling**, which rejects hidden objects/pixels earlier in the graphics pipeline so downstream stages do not spend work on output that will not be visible.
- A working-set / virtual-memory model in which the whitepaper describes up to **512 MB physical memory and 16 GB virtual memory** for a VPU-only system, with page-faulting allowing physical graphics memory to act like a cache.
- A **Vertex/Scalability Unit (VSU)** that can coordinate one or two VPUs, offers a full PCI Express x16 interface, and connects to each VPU over an interface rated at 4.2 GB/s in the whitepaper's system.
- **DirectBurst**: the host processor accumulates vertex data in a burst buffer and streams it to VSU DirectBurst memory. The whitepaper contrasts this with a conventional path requiring multiple main-memory accesses, and describes DirectBurst as requiring a single processor-to-main-memory read in its stated model.
- A **crossbar partitioner** for two VPUs. Rather than statically assigning one large screen half to each VPU, it distributes small checkerboard blocks so one VPU is less likely to receive all the detailed/heavy portion of a scene.
- **Staged floating-point precision**: 36-bit vertex values, 32-bit fragment processing and 16-bit floating-point outputs.
- **Genlock/framelock/ratelock** for synchronising displays and meeting a target frame period. Ratelock may discard a frame if the minimum requested period cannot be met.

Source PDF:
https://userpages.cs.umbc.edu/olano/s2004c01/ch13.pdf

These counts and memory figures are the whitepaper's statements about historical Wildcat systems, not specifications for modern Vulkan devices. The whitepaper's VPU/VSU configuration examples should not be assumed to describe every Realizm 800 board configuration identically.

## 2. Transferable ideas and Vulkan adaptations

| Historical technique | Why it mattered there | Proposed Vulkan/AI analogue | Testable workload |
|---|---|---|---|
| Hierarchical Z early culling | Avoids downstream processing for hidden pixels | **Work elision**: gate sparse/conditional work before expensive kernels; skip zero blocks, inactive experts or masked tokens when the saved work exceeds gating overhead | Sparse MoE, block-sparse matrices, attention masks |
| VSU crossbar + checkerboard partitioning | Spreads uneven screen workload among two VPUs | **Cost-balanced work partitioning**: schedule expert blocks, tiles or independent sequences by estimated work, not just equal item counts | MoE expert imbalance, variable-length batches, sparse workloads |
| DirectBurst host burst buffer | Reduces repeated host-memory accesses while preparing data for the graphics unit | **Streaming upload/ring-buffer path**: assemble metadata and weights in reusable pinned/staging buffers; minimise copies; feed transfer batches with explicit ownership | Expert paging, streamed weights, dispatch metadata |
| Unified virtual memory + page-fault caching | Allows working sets larger than physical graphics memory | **Explicit tensor residency manager**: hot/warm/cold tiers, page-in/out, async transfer, eviction and per-tensor access statistics | Models larger than VRAM, MoE experts |
| Long programmable shader programs | Supports complex algorithms in programmable hardware | **Typed virtual IR and cost-aware fusion**: merge operations when legal; split kernels if register pressure, occupancy or compiler time becomes worse | Fused activation/normalisation, attention/FFN subgraphs |
| Compiler-centric programmable pipeline | Avoids freezing evolving algorithms into fixed-function hardware | **Backend compiler passes** over a workload graph and virtual ISA, lowering to SPIR-V | All supported workloads |
| Multiple precision stages | Uses precision appropriate to each stage of the pipeline | **Precision policy**: quantised storage, lower-precision arithmetic where supported, FP32/integer accumulation, controlled conversion points | Q4/Q8 matvec, reductions, normalisation |
| 64-bit accumulation buffers (from Realizm product material, not established by this supplied whitepaper excerpt) | High-precision multipass graphics accumulation | Optional high-accuracy accumulation path for validation/reductions; not a default fast path | Long reductions, reference/accuracy testing |
| Genlock/framelock/ratelock | Coordinated frame timing and multi-display synchronisation | **Deadline-aware scheduling**: separate latency-sensitive token decode from throughput-oriented prefill; bound batch queues and define deadline/drop policies only where appropriate | Multi-session inference server |
| VPro command pipeline and lookup/imaging operations | Explicit stages and specialised graphics primitives | Typed V12-inspired operation semantics, lookup/transform nodes and command batching | Transform/lookup microbenchmarks |
| Kyro/Kyro II tile-based deferred rendering | Locality and tile-based work organisation reduce unnecessary external-memory work | Tile planner, local shared-memory reuse, fused tile kernels | GEMM, convolution, matrix-vector |
| Modern Vulkan compute | Executes real work on the user's current GPU | Independent direct GLSL/SPIR-V baseline; use the actual GPU's limits and features | All tests |

Important limit: Vulkan does not expose a general programmable crossbar between arbitrary compute engines, the historical VSU, or an equivalent page-fault mechanism on every device. Implement software analogues where practical and measure them. Do not imitate graphics-specific mechanisms literally where only the underlying scheduling/locality idea transfers.

## 3. New virtual IR / runtime components

Add these independently switchable components:

1. `WorkElisionPass`: detect inactive experts, all-zero blocks or provably unnecessary operations, preserving exact semantics.
2. `CostPartitioner`: split independent tiles/experts/sequences by estimated cost; measure predicted versus actual time.
3. `TransferRing`: reusable staging/upload ring buffers, avoiding per-dispatch allocation and duplicate CPU copies.
4. `ResidencyManager`: per-tensor residency state, priorities, last-use time, async transfers, eviction and page-in accounting.
5. `PrecisionPolicy`: declares storage, compute, accumulator and output types independently per operation.
6. `FusionCostModel`: weighs removed launches/intermediate writes against estimated register pressure, shader size and occupancy.
7. `DeadlineScheduler`: prioritises latency-sensitive decode vs bulk prefill under configurable service goals; always report fairness and throughput impacts.
8. `TilePlanner`: chooses tile sizes and layouts for the target Vulkan device, with autotuning rather than a fixed historical tile shape.

## 4. Benchmark variants

Use the same input tensors, outputs, precision semantics, device, synchronisation and timed region.

- `direct_simple`: straightforward direct Vulkan.
- `direct_optimised`: independent hand-tuned direct Vulkan implementation.
- `direct_tiled_fused`: direct shader with tiling and fusion.
- `vpu_fused`: virtual IR lowered to fused SPIR-V.
- `vpu_tiled`: virtual IR plus tile/locality pass.
- `vpu_work_elision`: work-elision pass enabled.
- `vpu_cost_partitioned`: cost-aware partitioning enabled.
- `vpu_streamed`: transfer-ring and residency strategy enabled for memory-pressure workloads.
- `vpu_precision_policy`: explicit per-stage precision choices.
- `vpu_deadline_scheduled`: server scheduling tests for decode/prefill concurrency.
- `vpu_combined`: autotuned combination of selected passes.

Every VPU variant must be compared to a strong direct Vulkan baseline. `vpu_combined` is not presumed best. Run ablations to determine which optimisation actually contributed.

## 5. Measurements and correctness requirements

- Compile shaders and create pipelines outside steady-state timing; report compile/pipeline time separately.
- Warm up each variant, alternate test order, use repeated batches and report median plus spread.
- Use Vulkan GPU timestamp queries for device execution and separate wall-clock timings for submission and end-to-end latency.
- Track dispatch count, bytes copied, estimated/observed memory traffic where available, peak resident bytes, page-in/page-out bytes, cache hit rate, queue wait and per-stage latency.
- Record GPU, driver, Vulkan version, compiler, workgroup shape, tile shape and relevant device limits.
- Check numerical tolerance and exact work equivalence; skipping work is only valid if it is provably irrelevant to the requested output.
- Include both microbenchmarks and complete LLM decode, prefill, and MoE traces.
- For memory paging, measure throughput and tail latency under a constrained VRAM budget, not only a fully-resident test.
- For scheduling, measure tokens/s, time-to-first-token, inter-token latency, throughput fairness and queueing delays.

## 6. Priority ranking

**P0 — do first**
1. Direct-vs-VPU benchmark hygiene and independent native baselines.
2. Fusion cost model, tile planner, and precision policy.
3. Transfer ring and telemetry for bytes moved / allocations / synchronisation.

**P1 — high value for large LLMs**
4. Residency manager and hot/warm/cold tensor paging.
5. Cost-partitioned scheduling for MoE experts and variable work.
6. Work elision for sparse or inactive computation, where semantic conditions are provable.

**P2 — workload-dependent**
7. Deadline scheduler for concurrent decode/prefill.
8. High-accuracy accumulation path.
9. More V12-inspired lookup/transform instructions if benchmarks show relevance to AI workloads.

## 7. Decision criterion

The architecture succeeds only if it improves the best direct-Vulkan result on relevant workloads, reduces memory requirements while maintaining usable throughput, or delivers another measurable service benefit. Each pass must be independently toggled, correct, profiled, and retained only when evidence justifies its complexity.

---

# v3 Research Additions: Dynamic Work Queues and GPU Algorithm Design

## 8. Stanford GPU Notes (2005–2007): asynchronous queues and dynamic work bins

Source: https://graphics.stanford.edu/~yoel/notes/index-0507.html

The notes argue that GPU programming need not rely exclusively on rigid multipass batches. Instead, queues, bins and write buffers can dynamically collect work into coherent bundles asynchronously. They describe a high-level scheduler that creates work bundles and a lower-level scheduler that executes them, and propose producer/consumer queues for coordinating conventional CPU cores and compute-oriented cores.

### Adaptation: VPU Queue/Bin Runtime

Add a scheduling layer above the virtual ISA. A work item should contain:

- operation/kernel family and shader pipeline key
- tensor/buffer identifiers and memory-residency requirements
- shape, datatype, quantisation and precision policy
- estimated cost and expected resource footprint
- dependency/fence state
- priority/deadline class
- compatibility key for batching with other work

The scheduler dynamically bins compatible work by kernel family, shape range, datatype and resource profile. It dispatches when a bin reaches a useful size, when a deadline approaches, or when waiting would leave the GPU underutilised. The policy must have a bounded wait so batching does not harm token latency.

### Portable implementation tiers

1. **CPU-managed dynamic queues (portable baseline):** CPU producers push jobs into bounded MPSC queues; the scheduler bins jobs and records Vulkan commands. Use reusable command buffers, staging rings, timeline semaphores and indirect dispatch where supported.
2. **GPU work queue inside a persistent dispatch (experimental):** a persistent shader claims pre-created task records with atomic counters and processes tasks until the queue is exhausted. This avoids requiring portable GPU-side creation of new Vulkan command buffers, but needs careful termination, occupancy and queue-contention design.
3. **Device-generated commands (optional capability path):** only when a supported Vulkan extension is detected. Never make this the required portable path.
4. **Hybrid CPU/GPU execution:** keep CPU work to tasks where CPU processing and hand-off beat the GPU alternative. For LLM inference, do not move logits or per-token routing state back to the CPU merely to imitate a heterogeneous architecture; that transfer can erase the gain.

### Vulkan and correctness caveats

- Standard Vulkan does not universally permit a compute shader to enqueue arbitrary new dispatch commands. GPU-produced indirect-dispatch parameters, extension-based device-generated commands, and persistent kernels are different mechanisms.
- Do not rely on a global workgroup barrier that Vulkan compute does not provide within a normal dispatch.
- Queue buffers require explicit memory dependencies and atomic operations; task publication must be visible before consumers claim tasks.
- Queueing is useful only when enough independent work exists. Small queues can add atomics, CPU scheduler overhead and latency.
- Queue scheduling and static batching must be benchmarked with identical workload semantics.

### New benchmark variants

- `vpu_static_batch`: preselected fixed batches.
- `vpu_cpu_workqueue`: CPU-managed work bins and dynamic batch formation.
- `vpu_persistent_queue`: shader-resident task queue, when supported by an experimental kernel.
- `vpu_hybrid_queue`: CPU and GPU workers share the scheduler where workload transfer economics justify it.

Measure throughput, task wait time, queue depth, GPU idle gaps, bin fill time, atomics per task, tail latency, time-to-first-token and inter-token latency. Test uniform tasks as well as intentionally skewed workloads, including one or two overloaded MoE experts.

## 9. Ivo Hanák (2005): The GPU and Graphic Algorithms

Source: https://holo.zcu.cz/txt/TR-2005-05.pdf
Alternative historical path referenced in its citations: http://www.kiv.zcu.cz/publications/2005/tr-2005-05.pdf

The indexed abstract describes the report as an overview of programmable graphics hardware, its applications and proposed improvements, illustrated with selected algorithm examples. Direct access to the PDF timed out during this review, so detailed claims about particular algorithms in the report are intentionally not asserted here.

### How to use this source in the project

Treat this as a survey of algorithm-to-GPU mapping rather than a hardware ISA specification. Review its examples, when a full-text copy is available, for:
- how non-graphics algorithms are reformulated as many parallel independent operations
- where graphics-pipeline stages and data representation force extra passes
- how intermediate results can remain in GPU-resident resources instead of being read back to the CPU
- how algorithmic complexity, control divergence and data layout interact with GPU efficiency
- which graphics-specific tricks become unnecessary or can be replaced with native Vulkan compute

The project should extract each technique into a hypothesis with a modern compute counterpart and a measurable test. Because the report dates from 2005, any API-specific advice must be translated into current Vulkan concepts; do not copy legacy OpenGL/Cg limitations as if they applied to Vulkan.

## 10. Combined scheduling model

The three architectures/lines of work contribute different layers:

1. **VPro / Realizm-inspired operation and pipeline concepts:** typed operation IR, command preparation, explicit precision and resource policy.
2. **Kyro-inspired locality:** tiles, reusable working sets, reduced redundant memory traffic.
3. **Realizm-inspired load balancing and paging:** cost-aware partitioning, residency, transfer rings and per-stage precision.
4. **Stanford queue/bins model:** dynamic work collection, producer-consumer dependencies and hybrid CPU/GPU scheduling.
5. **Modern Vulkan backend:** compiled SPIR-V, independent native baselines, device-capability checks and measured target-specific kernels.

Do not combine every technique by default. The optimiser should choose passes per workload and keep all passes switchable to support ablation.

## 11. Queue-oriented scheduler outline

```text
CPU / IO producers
  ├─ tensor page-in completed
  ├─ route/shape metadata ready
  ├─ decode job ready
  └─ prefill / expert jobs ready
              |
              v
      bounded producer queues
              |
              v
      compatibility binning
      ├─ kernel family
      ├─ shape / tile class
      ├─ datatype / quant format
      ├─ memory residency
      └─ priority / deadline
              |
              v
      cost + deadline policy
      ├─ wait briefly to fill batch
      ├─ flush if latency deadline nears
      ├─ coalesce adjacent transfers
      └─ prioritise residency-ready work
              |
              v
       Vulkan command batches
              |
              v
     timeline-semaphore dependencies
              |
              v
       native Vulkan execution
              |
              v
       completion / telemetry
```

This is a scheduler model, not a guarantee that a GPU shader can freely launch another shader. Keep command generation on the CPU for the portable first version.

## 12. New implementation priorities

**P0**
1. Keep benchmark timing fair and log actual GPU/driver/compiler configuration.
2. Add CPU-managed dynamic queues and compatible-work binning.
3. Track queue wait, batch fill time, GPU idle gaps and transfer bytes.

**P1**
4. Test queue-based scheduling on uneven MoE expert workloads and mixed decode/prefill.
5. Add transfer ring and weight/tensor residency manager.
6. Compare bounded-wait dynamic bins with fixed batching to quantify the latency-throughput trade-off.

**P2**
7. Prototype a persistent GPU task queue for independent, uniform task records.
8. Add optional device-generated commands only on Vulkan devices that explicitly support the required extension.
9. Revisit the Hanák report at full text when available and extract more source-specific techniques.

## 13. Decision criterion

Queue-based VPU execution earns its complexity only if it improves utilisation or tail latency on a representative dynamic workload without regressing steady-state decode. The VPU as a whole should be compared against the best independently tuned direct-Vulkan path, and every scheduling, fusion, tiling, paging and precision feature must be measurable in isolation.

---

# v4 Research Additions: Cobalt, Kyro II, and Chromium stream-processing thesis

## 14. SGI Cobalt: integrated graphics pipeline and shared memory

Cobalt was the graphics/system ASIC in the SGI Visual Workstations 320 and 540. The SGI graphics white paper says it integrated geometry and rasterisation pipelines, a memory controller, and interfaces to the CPU, memory, I/O and display subsystems. It describes one dynamically allocated graphics-memory pool drawn from main memory: up to 1 GB on the 320 and up to 2 GB on the 540. The system memory pool could hold framebuffer, depth, texture, rendering, imaging and video data. The paper reports a 3.2 GB/s graphics-to-memory path in its historical configuration.

Primary source:
- SGI, *The Silicon Graphics 320 and 540 Visual Workstation Graphics Subsystem*: https://sgi320.wordpress.com/wp-content/uploads/2013/01/320_gfx_subsystem.pdf
- Linux kernel note, SGI Visual Workstations: https://kernel.googlesource.com/pub/scm/linux/kernel/git/jic23/iio/%2B/1da177e4c3f41524e886b7f1b8a0c1fc7321cac2/Documentation/sgi-visws.txt

The user's Cobalt capability list additionally includes primitive rasterisation and setup, culling and clipping, Gouraud shading and lighting, multi-format colour/depth/stencil buffers, texture filtering, anti-aliased lines, fog, alpha/chroma tests, blending, logic operations, pixel-format conversion, 4x4 colour-matrix transforms, YCrCb/RGB conversion, fast buffer copies, off-screen buffers, statistics/occlusion tests and frame-lock synchronisation.

### Transferable Cobalt ideas

| Cobalt feature | VPU/Vulkan adaptation | Relevant benchmark | Important limitation |
|---|---|---|---|
| Unified system/graphics memory pool | Memory-backend abstraction: device-local VRAM, host-visible memory, shared UMA memory where supported; track actual placement and residency | Models/tensors exceeding VRAM; transfer and paging overhead | Shared addressability is not the same as equal latency/bandwidth. On a discrete GPU, host-visible system RAM can be much slower than VRAM. |
| Hardware buffer allocation and fast buffer copy | Persistent buffer allocator, transfer ring, copy batching, and explicit ownership/dependency tracking | Weight paging, activation staging, KV-cache compaction | Copy queues only help if copies overlap usefully with execution. |
| Off-screen/p-buffers | Keep intermediate tensors and scratch resources device-resident across graph stages | Fused and multi-stage image/AI pipelines | Use storage buffers/images according to access pattern; do not emulate legacy framebuffer objects literally. |
| 4x4 colour matrix, component swizzle and scale/bias | Fused affine transform / normalisation / image pre- and post-processing op | Vision preprocessing, data conversion, transform+dot microbench | This is a natural AI-adjacent Cobalt operation, but it will not accelerate large GEMM by itself. |
| YCrCb ↔ RGB and pixel-format conversion | Optional fused video/image preprocessing kernels, reducing intermediate buffers and launches | Video inference inputs and output conversion | Relevant for vision/video pipelines, not most text-only LLM execution. |
| Texture filtering (nearest/bilinear/trilinear) | Lookup/interpolation primitives where interpolation is the algorithm | Resampling, image pyramids, continuous lookup tables | Do not substitute texture sampling for dense matrix arithmetic. |
| Occlusion/statistics instrumentation | General runtime telemetry; use validity masks/counters to skip work only when semantically safe | Sparse/conditional AI workloads | Graphics occlusion semantics do not transfer directly to model pruning. |
| Connected meshes and lower command traffic | Compact work descriptors and batch submission | Small kernels, expert routing and per-operation dispatch overhead | Avoid batching so long that latency-critical work waits unnecessarily. |
| Depth/stencil, alpha/chroma, dithering, overlays, line stippling and frame lock | Preserve as graphics compatibility only, or use applicable principles in vision/video utilities | Graphics and image workloads | Mostly graphics-specific; not priorities for an LLM compute engine. |

**Memory-size note:** the 2 GB figure is documented for the SGI 540 system configuration; the same white paper describes up to 1 GB on the 320. Do not treat either historical number as a limit on a modern VPU virtual address space.

## 15. PowerVR3 STG4500 Kyro II: deferred visibility, DLR and tile-local work

The supplied Kyro II notes identify a 32×16-pixel tile size, a tile-based deferred rendering architecture, two pixel pipelines with independent texture units, multiple texture layers, filtering including anisotropic sampling, and a Display List Renderer (DLR) that divides scene work into sections. The key architectural idea is not the exact tile dimensions; it is collecting/bucketing work by region before spending the more expensive per-pixel work.

PowerVR's technical descriptions explain the general tile-based deferred rendering principle: geometry is assigned to tiles; hidden surfaces are rejected before deferred shading where possible; per-tile working data such as depth/colour can remain local, reducing unnecessary external-memory traffic. That principle is documented for the architecture family, not a claim that Vulkan exposes Kyro's proprietary hardware blocks.

Sources:
- Imagination Technologies, Tile-Based Deferred Rendering: https://docs.imgtec.com/starter-guides/powervr-architecture/html/topics/tile-based-deferred-rendering-index.html
- PowerVR Kyro overview, AnandTech: https://www.anandtech.com/show/558/4
- Supplied specifications in this project: Kyro II STG4500 notes

### Transferable Kyro II ideas

| Kyro II feature | VPU/Vulkan adaptation | Test |
|---|---|---|
| 32×16 tile rendering | Start with several compute tile candidates, not a single fixed shape; choose per kernel/device with autotuning | GEMM, convolution, matrix-vector and local reductions |
| Deferred texturing until hidden work is rejected | Work-elision pass that avoids provably inactive computations before expensive stages | Sparse blocks, zero experts, masked operations, early-exit paths |
| DLR / scene sections and per-tile display lists | Work-bin builder: group independent tensor tiles or expert jobs by destination, shape, datatype and pipeline key | Irregular expert sizes and variable-sequence prefill |
| Two pixel pipelines, each with texture capability | Pipeline-level concurrency and multiple independent work queues where they improve occupancy | Several independent small kernels; avoid manual subdivision of one small kernel unless profiling supports it |
| Up to eight texture layers blended in one pass | Multi-operation fusion, while retaining separate intermediate values only when required | Transform/scale/bias/activation chains and convolution epilogues |
| Bilinear/trilinear and anisotropic sampling | Explicit interpolation/lookup ISA operation for sampling workloads | Vision preprocessing, resampling and lookup-table experiments |
| No dedicated hardware T&L block | Avoid assuming preprocessing is free; separately measure CPU-side work, compute preprocessing and fused preprocessing | Input preparation and small-workload launch overhead |
| Full-screen anti-aliasing without a larger framebuffer | Reuse/intermediate-storage strategy that does not require a full duplicate output surface where an algorithm allows it | Image filters and scratch-memory footprint, not general LLM acceleration |
| Texture compression and bump-mapping variants | Compressed/quantised data-path experiments only where decode and arithmetic economics justify them | Quantised vision tensors and packed lookup data |

**Important translation rule:** Kyro's hidden-surface removal is visibility-aware graphics processing, not generic sparse-matrix acceleration. For AI, skip work only when masks, routing, sparsity or data prove that the work cannot affect the output.

## 16. Greg Humphreys' thesis: Chromium and stream-processing graphics

Source: Greg Humphreys, *A Stream Processing Approach to Interactive Graphics on Clusters of Workstations*, Stanford PhD dissertation, July 2002:
https://graphics.stanford.edu/papers/humper_thesis/humper_thesis.pdf

The thesis describes Chromium as a system for transforming streams of standard graphics API commands using configurable Stream Processing Units (SPUs). It details SPU chains, stream serialisation, state tracking, geometry bucketing, lazy state updates, context differences, and sort-first/sort-last parallel rendering. The `tilesort` SPU routes only the necessary geometry to relevant display regions. The thesis also points out trade-offs: bucketing has bookkeeping cost and depends on spatial locality; sort-first can be hard to load-balance; sort-last readback/compositing can become a bottleneck.

### Transferable Chromium ideas

| Thesis concept | VPU/compiler analogue | Implementation |
|---|---|---|
| Programmable SPU chain | Pass pipeline over a typed virtual IR | Compose validation, fusion, tiling, layout, paging and scheduling passes |
| Stream transformations separated from renderer | Backend-independent workload graph lowered to Vulkan | Keep virtual ISA semantics independent of GLSL and SPIR-V details |
| Geometry bucketing | Bucket tensor tiles/jobs by compatible execution properties | Build bins by kernel family, shape bucket, datatype, quant scheme and resource requirements |
| Lazy state update | Defer pipeline/binding/resource changes until a dispatch actually needs them | Dirty-bit tracking and differential pipeline/descriptor binding updates |
| Context difference computation | Compare desired dispatch state to current recorded state and emit only necessary changes | State cache for pipelines, descriptors, push constants and specialization variants |
| Packed geometry buffers and batching | Packed dispatch descriptors and staging/command arenas | Amortise metadata processing, allocation and queue submission across many tasks |
| Sort-first partitioning | Partition work before kernel execution by output/data tile | Useful when tiles have clear ownership and limited overlap |
| Sort-last parallel work and final composition | Parallel partial results followed by reduction/combine | Suitable for mathematically valid reductions; avoid full tensor readback to CPU |
| SPU inheritance/composable filters | Reusable compiler/runtime passes with declared inputs, outputs and invariants | Independent plugins/pass interfaces with capability checks |
| Explicit synchronisation between command streams | Dependency DAG and Vulkan timeline semaphores/fences/barriers | Preserve correctness across transfer, compute and output-readback stages |

### A key lesson from the thesis

Do not rebucket every operation individually if bucketing itself costs too much. The thesis amortises spatial-bound calculations by packing many primitives, and flushes when buffers fill or state requires it. The AI version should likewise amortise job classification over groups of operations but flush batches when deadlines, dependency changes, residency or available parallelism demand it.

## 17. Updated runtime architecture

```text
Model graph / inference requests
              |
              v
Typed virtual IR (backend independent)
              |
              v
Composable pass chain
  ├─ validate and canonicalise
  ├─ lazy state / redundant binding elimination
  ├─ work elision where semantically provable
  ├─ cost and shape bucketing
  ├─ tile/layout/autotune
  ├─ fusion cost model
  ├─ precision policy
  └─ residency and transfer planning
              |
              v
Dynamic bounded work bins
  ├─ decode priority/deadline
  ├─ prefill batches
  ├─ MoE expert jobs
  ├─ vision/preprocessing
  └─ paging/copy jobs
              |
              v
Vulkan command recorder + transfer ring
              |
              v
SPIR-V pipelines / indirect dispatch if supported
              |
              v
Device execution; explicit dependencies
              |
              v
GPU timestamps + telemetry + correctness checks
              |
              v
Autotuner feedback, per-device cache and report
```

The portable first version should create/record Vulkan commands on the CPU. GPU-resident task queues or device-generated commands are optional paths gated by actual feature support, not assumptions.

## 18. New benchmark variants and ablations

Add the following variants to the existing direct-vs-VPU suite:

1. `direct_reference`: simple independent Vulkan baseline.
2. `direct_optimised`: tuned native baseline.
3. `vpu_static_passes`: fixed pass chain, no dynamic binning.
4. `vpu_stream_passes`: composable IR pass chain.
5. `vpu_lazy_state`: cached desired/current pipeline and descriptor state.
6. `vpu_binned`: jobs batched by kernel/shape/type/residency.
7. `vpu_bounded_queue`: dynamic bins with bounded waiting and deadlines.
8. `vpu_tiled`: tile/local-memory optimiser.
9. `vpu_work_elision`: sparse/conditional skip pass with correctness proof.
10. `vpu_resident`: persistent tensor residency and staging/copy ring.
11. `vpu_combined`: optimiser-selected combination of the preceding passes.

For each workload, compare against the fastest *measured* direct implementation as well as a semantics-matched shader. Record compilation time separately; GPU time with timestamp queries; host submission and end-to-end time; dispatch count; buffer-copy bytes; queue wait/batch fill; resident memory; paging bytes; and numerical tolerance. Repeat with warm-up and report median plus spread.

## 19. Priority after these sources

**P0: trustworthy evaluation**
- Make GPU identity, driver, compiler, timestamps and repetitions part of every report.
- Ensure direct reference kernels are independently implemented and tuned rather than emitted from the VPU's shader template.
- Compare equivalent work and outputs; fix any workload/baseline mismatches before interpreting speedups.

**P1: high-leverage compiler/runtime structure**
- Typed backend-independent IR and composable pass interface.
- Lazy state / differential descriptor and pipeline binding.
- Packed dispatch descriptors, transfer rings and bounded dynamic work bins.
- Cost-aware fusion + tile planning.

**P2: capacity and irregular workloads**
- Tensor/weight residency manager with hot/warm/cold tiers.
- Cost-balanced work partitioning for uneven MoE experts and variable sequence lengths.
- Work-elision on provably inactive blocks/experts.
- Optional indirect or GPU-resident queues only after portable CPU-managed queues are measured.

**P3: specialized vision/data operations**
- Cobalt-like 4×4 colour transform and YCrCb/RGB conversion.
- Interpolation and sampling operations inspired by historical texture units.
- Retain alpha/stencil/dither/frame-lock behaviour only if graphics compatibility is an explicit goal.

## 20. Final decision criterion

The VPU is useful only if it beats the best direct Vulkan implementation on representative workloads, or provides a measured capability such as bounded-memory model execution or improved tail latency. Historical architectures should contribute modular, switchable design ideas; they should not impose a historical instruction set or graphics pipeline on all AI work.
