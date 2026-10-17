# OpenBLAS → VPU → OpenVINO: Function Mapping Table

**Purpose.** Map every OpenBLAS function family through the V12 method catalogue
(FUSE, ELIDE, REDUCE, PRECISION, TILE, COST-GATE) onto the Intel GPU backend
(Arc iGPU, Xe-LPG) via OpenVINO IR, using the Cell-derived staging and
placement principles from `VPU_CELL_CONCEPTS_v0.1.md` (P3, H-A through H-H).

**Hardware target.** Intel Core Ultra 5 135H: P-cores (Raptor Cove, AVX2 +
AVX-VNNI), E-cores (LP E-cores, no AVX-VNNI), Arc iGPU (Xe-LPG, 128 EU, SLM per
Xe-core queried at runtime per §2 of Cell concepts).

**Baseline.** OpenBLAS x86_64 kernels compiled with AVX2 (SSE4.1/SSE3 as fallback)
on CPU; OpenVINO GPU plugin for offload. R6: direct baseline is the best
hand-tuned OpenBLAS kernel at equal parallelism, not a weaker variant.

**V12 method key:**

| Tag | Mechanism | What it does for the GPU |
|---|---|---|
| V12.FUSE | N ops → 1 dispatch | Merge elementwise chains into single OpenCL/OneAPI kernel; merge GEMV+activation (as in main.cpp:V12.FUSE ablation) |
| V12.ELIDE | skip provably-inactive work | Skip zeroed rows in sparse formats, skip upper/lower triangle of symmetric matrices, skip inactive MoE experts |
| V12.REDUCE | subgroup-parallel reduction | Horizontal dot-product sums, norms, reductions within a subgroup — maps to Xe subgroup operations |
| V12.PRECISION | per-stage type selection | INT8 weights with FP32 accumulation (matches main.cpp:V12.PRECISION), BF16, or FP16 |
| V12.TILE | autotuned tile/locality | Workgroup tile size for GEMM/GEMV, shared local memory (SLM) staging per P3/H-A |
| V12.COST-GATE | cost model selects rewrites | Decide fusion vs split, tile size vs register pressure, precision per op |

---

## SIMD Target Inventory (x86-64)

**We need (host-side OpenBLAS baseline on Ultra 5 1355H):**

| SIMD | Present? | Where used | Action |
|---|---|---|---|
| SSE | Yes (baseline x86-64) | `common.h`, `intrin_sse.h` | Keep as fallback |
| SSE2 | Yes (baseline x86-64) | `common.h`, `intrin_sse.h` | Keep as fallback |
| SSE3/SSSE3/SSE4.1/4.2 | Yes | `intrin_sse.h`, haswell/sandybridge kernels | Enable via `HAVE_SSE3`/`HAVE_SSE4_1` |
| AVX | Yes | `intrin_avx.h` | Enable via `HAVE_AVX` |
| AVX2 | Yes (Raptor Cove) | `intrin_avx.h`, skylakex kernels | **Primary CPU SIMD target** |
| AVX-VNNI | Yes (Meteor Lake) | Not yet in this source tree | Add `HAVE_VNNI` flag for INT8 dotprod |
| FMA3 | Yes (Raptor Cove) | `intrin_sse.h`, `intrin_avx.h` | Enable via `HAVE_FMA3` |
| AMX | No (mobile SKU) | `cpuid.h` line 149 | **Do NOT compile** — guard with `HAVE_AMXBF16` check at runtime |
| AVX-512 | No (mobile SKU) | `intrin_avx512.h`, skylakex kernels | **Do NOT compile** — guard out for Xe-LPG target |
| WASM SIMD | No | `intrin_wasm.h` | **Do NOT compile** unless targeting web |

**We do NOT need:**
- `kernel/simd/intrin_avx512.h` — only for SKYLAKEX/COOPERLAKE (server); skip on mobile
- `kernel/x86_64/*_skylakex.*` — AVX-512; not applicable to Meteor Lake
- `kernel/x86_64/amx*` — AMX is server-only
- `kernel/simd/intrin_wasm.h` — browser target
- ARM/NEON, POWER/Altivec, RISC-V vector, MIPS SIMD — wrong architecture

---

## Level 1 BLAS: Vector-Vector operations

### Reductions (V12.REDUCE)

| OpenBLAS fn | V12 method | OpenVINO mapping | classification | feas | verify | risk | doc |
|---|---|---|---|---|---|---|---|
| `sdot`/`ddot` | V12.REDUCE | `ov::op::v0::ReduceOp` fused with subgroup reduce in custom OpenCL kernel | adapt | green | r2 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.1 |
| `scnrm2`/`dnrm2`/`scnrm2` | V12.REDUCE + V12.TILE | One-pass squared-sum reduction in SLM, sqrt at end | adapt | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.1 |
| `sasum`/`dasum`/`scasum`/`dzasum` | V12.REDUCE | Horizontal sum via subgroup shuffle, one workgroup per 256 elements | adapt | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.1 |
| `isamax`/`idamax`/`icamax`/`izamax` | V12.REDUCE | Parallel max + argmax in shared/local memory; ties per BLAS spec | adapt | green | r3 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.1 |
| `sdot`/`ddot` (batching) | V12.FUSE + V12.REDUCE | Batch N dot-products into one kernel; each workgroup computes one dot via subgroup reduce | adapt | green | r4 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.1 |

**Debug-hypothesis block (ddot):** At O(N=65536) per thread, OpenBLAS x86_64 has
`N/4` FMA chains then horizontal adds. Xe-LPG subgroup (16 lanes, 512 threads
per EU) can hold partial accumulator per lane; V12.REDUCE cuts memory
round-trips from O(N) to O(N/8) with 8x subgroup reduce. Hypothesis: GPU beats
CPU only when dispatch overhead < kernel residency; if `<1024` elements, dispatch
cost dominates → fall back to CPU (V12.COST-GATE).

### Elementwise (V12.FUSE candidates)

| OpenBLAS fn | V12 method | OpenVINO mapping | classification | feas | verify | risk | doc |
|---|---|---|---|---|---|---|
| `saxpy`/`daxpy` | V12.FUSE | Direct elementwise multiply-add; batch as V12.FUSE if fused with downstream | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.2 |
| `scopy`/`dcopy` | V12.FUSE | Memory copy via OpenVINO `v0::ReadValue`/`Assign` or staged transfer (H-A ring) | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.2 |
| `sswap`/`dswap` | V12.FUSE | In-place swap; batch multiple swaps if pattern known | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.2 |
| `sscal`/`dscal` | V12.FUSE | Elementwise multiply; fuse with consumer | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.2 |
| `srot`/`drot` | V12.FUSE | Givens rotation: c*x + s*y, -s*x + c*y in one fused pass | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.2 |
| `srotg`/`drotg`/`srotm`/`drotmg` | none | Scalar setup; stays on host | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.2 |

**Debug-hypothesis block (axpy+gemv fusion):**
`cgemv → y = alpha*A*x + beta*y` has a saxpy epilogue. V12.FUSE merges the
gemv and the axpy into a single kernel. OpenBLAS already does this on CPU
(see `gemv_n_atom.S` vs `gemv_n.c`). On GPU, one dispatch vs two saves
`~2 × dispatch_overhead (4–8 µs)`. Hypothesis: fusion pays when
matrix ≥ 64×64; below that, separate kernels have lower register pressure.

---

## Level 2 BLAS: Matrix-Vector operations

### GEMV family (V12.TILE + V12.REDUCE + V12.PRECISION)

| OpenBLAS fn | V12 method | OpenVINO mapping | classification | feas | verify | risk | doc |
|---|---|---|---|---|---|---|---|
| `sgemv` (no-transpose) | V12.TILE + V12.REDUCE | Workgroup tiles K×4; SLM staging of x vector (P3); subgroup reduce over K | adapt | green | r2 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `dgemv` (no-transpose) | V12.TILE + V12.REDUCE | Same tiling; accumulate in FP32 subgroup shuffles | adapt | green | r2 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `sgemv` (transpose) | V12.TILE + V12.REDUCE | Transpose case: each workgroup produces one output element; reduce over M | adapt | green | r2 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `cgemv`/`zgemv` | V12.TILE + V12.REDUCE | Complex Givens rotations; subgroup complex-add reduce | adapt | yellow | r3 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `sgbmv`/`dgbmv` | V12.ELIDE + V12.TILE | Banded: skip out-of-band zero entries (H-G alignment reject); tile over band | adapt | green | r4 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `ssbmv`/`dsbmv` | V12.ELIDE + V12.PRECISION | Symmetric band: compute only non-zero band (skip zeros) | adapt | green | r4 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `ssymv`/`dsymv` | V12.ELIDE (symmetry) + V12.TILE | Only lower/upper triangle → half the work; tile over non-zero region | adapt | green | r5 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `strmv`/`dtrmv` | V12.TILE + V12.ELIDE | Triangular: skip upper/lower zero triangle; workgroup tile | adapt | green | r5 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `strsv`/`dtrsv` | V12.TILE + V12.REDUCE (forward/back sub) | Triangular solve: forward substitution is sequential across panels → split into panel-batch (V12.FUSE with dependency) | rethink | yellow | r6 | high | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `sger`/`dger` | V12.TILE + V12.FUSE | Rank-1 update: y += alpha*x*y^T; tile over M×N; fuse add+write | adapt | green | r6 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `ssyr`/`dsyr` | V12.ELIDE (symmetry) + V12.TILE | Symmetric rank-1: compute upper triangle only, mirror | adapt | green | r7 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `ssyr2`/`dsyr2` | V12.ELIDE + V12.TILE | Symmetric rank-2: compute upper triangle only | adapt | green | r7 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `spmv`/`dpmpv` | V12.ELIDE (packed symmetry) | Packed symmetric: load packed, unpack on-fly, exploit symmetry | rethink | yellow | r8 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `sspr`/`dspr` | V12.ELIDE + V12.PRECISION | Packed symmetric rank-1: half-write only; INT8 if scale allows | adapt | green | r8 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `stpmv`/`dtpmv` | V12.TILE + V12.ELIDE | Packed triangular: decode indices on-fly; skip zeros | adapt | yellow | r9 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |
| `stpsv`/`dtpsv` | V12.TILE + V12.REDUCE | Packed triangular solve: forward sub with packed decode | rethink | yellow | r10 | high | docs/OpenBLAS-VPU-20261015-2031.md §4.3 |

**Debug-hypothesis block (sgemv T vs N):**
- No-transpose: M outputs, each reads K elements. N×K matrix fits in SLM if
  `N*K*4 < SLM_size`. Tile: each workgroup computes 32 outputs, loads 32×K
  submatrix into SLM, reduces over K via subgroup. 
- Transpose: N outputs, each reads M elements. M×K matrix transposed = K×M.
  Tile: each workgroup produces one output, loads M elements, reduces.
- Hypothesis: transpose variant has worse reuse (no data sharing between outputs).
  Tiled N-transpose wins when M×K > 8192×8192 (exceeds L2, must use SLM).

---

## Level 3 BLAS: Matrix-Matrix operations

### GEMM family

| OpenBLAS fn | V12 method | OpenVINO mapping | classification | feas | verify | risk | doc |
|---|---|---|---|---|---|---|---|
| `sgemm` | V12.TILE + V12.FUSE | OpenVINO `ov::op::v0::MatMul` with INT8/FP32; cost-gate tile size via autotuner | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `dgemm` | V12.TILE + V12.PRECISION | FP64 MatMul; Xe-LPG supports FP64 at 1/16 throughput → consider FP32 emulation | adapt | yellow | r2 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `cgemm`/`zgemm` | V12.TILE + V12.FUSE | Complex MatMul; decompose into 4 real MatMuls, fuse accumulation | adapt | yellow | r3 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `gemm_batch` | V12.FUSE | Batch → single large MatMul with batched dim; or group small ones (P16) | adapt | green | r4 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `gemm_batch_strided` | V12.FUSE + V12.TILE | Strided batch: map to OpenVINO batched MatMul; tile batch dim | direct | green | r4 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `sbgemm` | V12.PRECISION | BF16 compute; Xe-LPG supports via oneDNN BF16 path | direct | green | r5 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `sbgemm`/`shgemm` | V12.PRECISION | BF16/F16 compute with FP32 accumulate | direct | green | r5 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `sgemm3m`/`zgemm3m` | V12.FUSE | 3m complex GEMM: fuse 3 real GEMMs into one kernel | adapt | yellow | r6 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |

**Debug-hypothesis block (sgemm tiling):**
OpenBLAS x86_64 uses 8×8 or 16×4 microkernels with register blocking tuned to
L1 (32KB). Xe-LPG shared local memory (SLM) per Xe-core is queryable
(§2 of Cell concepts — do NOT assume size). Hypothesis:
- Tile `M_tile × N_tile = 128 × 128` with `K_chunk=64` uses ~64KB SLM (fits if
  SLM ≥ 64KB; if only 32KB, tile down to 64×64).
- V12.TILE: if matrix fits in L2, skip SLM copy (no reuse benefit). Only tile
  when `M×N×4 > 1MB` (exceeds L2 per slice).
- V12.PRECISION: for LLM inference, INT8 weights + FP32 accumulation is within
  1e-2 tolerance (matches main.cpp baseline). FP16 saves 2× bandwidth,
  increases overflow risk.

### Symmetric / Triangular Level 3

| OpenBLAS fn | V12 method | OpenVINO mapping | classification | feas | verify | risk | doc |
|---|---|---|---|---|---|---|---|
| `ssymm`/`dsymm` | V12.ELIDE (symmetry) | Compute only lower or upper triangle of A | adapt | green | r7 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `ssyrk`/`dsyrk` | V12.ELIDE + V12.TILE | Symmetric rank-k: compute only lower/upper triangle of C, tile | adapt | green | r7 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `ssyr2k`/`dsyr2k` | V12.ELIDE + V12.TILE | Symmetric rank-2k: compute only one triangle; fuse two GEMMs | adapt | green | r8 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `chemm`/`zhemm` | V12.ELIDE (Hermitian) | Hermitian: conjugate-transpose symmetry; compute half | adapt | yellow | r8 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `cherk`/`zherk` | V12.ELIDE + V12.TILE | Hermitian rank-k: one-triangle compute, conjugate | adapt | yellow | r9 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `cher2k`/`zher2k` | V12.ELIDE + V12.TILE | Hermitian rank-2k: fuse two conjugate transposed products | adapt | yellow | r9 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `strmm`/`dtrmm` | V12.ELIDE + V12.TILE | Triangular multiply: skip zero triangle; tile | adapt | green | r10 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `strsm`/`dtrsm` | V12.TILE + V12.REDUCE (forward sub) | Forward substitution is inherently sequential; tile into panels; V12.COST-GATE decides panel size | rethink | yellow | r11 | high | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |
| `sgemmt` | V12.ELIDE + V12.TILE | Full output triangle only; same as syrk but with explicit trans flags | adapt | green | r10 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.4 |

**Debug-hypothesis block (trsm):**
Triangular solve X ← A^{-1}B is data-dependent: row i depends on rows 0..i-1.
On CPU, OpenBLAS uses blocked algorithm (level-3 BLAS inside level-3) with
recursive panel factorization. On GPU, each panel solve is sequential along the
diagonal. Hypothesis: tile into 64×64 panels; within each panel, forward sub
along diagonal is sequential but across panels of the same block-row, can be
parallel (V12.REDUCE on partial solutions). Cost gate: below 64×64, dispatch
overhead > solve time.

---

## Memory and Layout Extensions

| OpenBLAS fn | V12 method | OpenVINO mapping | classification | feas | verify | risk | doc |
|---|---|---|---|---|---|---|---|
| `somatcopy`/`domatcopy` | V12.TILE + V12.FUSE | Matrix transpose/copy; tile to SLM to avoid bank conflicts | adapt | green | r12 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.5 |
| `imatcopy`/`simatcopy` | V12.TILE + V12.FUSE | In-place transpose; SLM buffer needed (no in-place on GPU) | adapt | green | r12 | med | docs/OpenBLAS-VPU-20261015-2031.md §4.5 |
| `geadd` | V12.FUSE + V12.TILE | C += alpha*A + beta*B; fuse into single elementwise kernel | direct | green | r12 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.5 |
| `set_matrix`/`get_matrix` | V12.TILE + H-A | Bulk memory transfer via H-A staging ring (P15) | adapt | green | r13 | low | docs/OpenBLAS-VPU-20261015-2031.md §4.5 |

**Debug-hypothesis block (imatcopy in-place):**
OpenBLAS imatcopy can do in-place transpose on CPU (with careful stride handling).
GPU cannot do true in-place transpose without extra memory. Hypothesis:
allocate scratch SLM, transpose into it, then copy back → 2× memory but 1
dispatch. V12.TILE: use 32×32 tile in SLM; 8 banks means stride-1 access
pattern avoids bank conflicts when tiles are 32-wide (32/8 banks = 4-way
interleaved).

---

## OpenBLAS Thread Control and Configuration

| OpenBLAS fn | V12 method | OpenVINO mapping | classification | feas | verify | risk | doc |
|---|---|---|---|---|---|---|---|
| `openblas_set_num_threads` | V12.COST-GATE | Replace with OpenVINO `ov::threading` hint or GPU thread count | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §5 |
| `openblas_get_num_procs` | none | Query `maxComputeWorkgroupsInX` + EU count from OpenVINO device | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §5 |
| `openblas_get_config` | none | Map to `DeviceInfo` in context.hpp:112 (OpenVINO device query) | direct | green | r1 | low | docs/OpenBLAS-VPU-20261015-2031.md §5 |
| `openblas_setaffinity` | V12.COST-GATE | CPU affinity → GPU: dispatch order sweep (H-C) rather than affinity | adapt | green | r14 | low | docs/OpenBLAS-VPU-20261015-2031.md §5 |

---

## Summary by V12 method

### Which OpenBLAS functions benefit most from each V12 method?

**V12.FUSE** (N ops → 1 dispatch):
- `sgemv` + bias/activation epilogue → fused as in moe_layer.cpp:134 (SiLU activation)
- `sger`/`dger` rank-1 update → fuse outer product + add
- `ssyr`/`dsyr` → fuse symmetric write
- `gemm_batch` → group into single large dispatch
- All Level 1 elementwise ops (`saxpy`, `sscal`, `scopy`, `sswap`, `srot`)

**V12.ELIDE** (skip provably-inactive work):
- All symmetric (`ssymv`, `ssyr`, `ssyr2`, `ssymm`, `ssyrk`, `ssyr2k`, `chemm`, `cherk`)
- All triangular (`strmv`, `strsv`, `strmm`, `strsm`, `tbmv`, `tbsv`, `tpmv`, `tpsv`)
- Banded (`sgbmv`, `ssbmv`) → skip out-of-band zeros
- Packed symmetric (`spmv`/`dpmpv`) → skip redundant triangle

**V12.REDUCE** (subgroup-parallel reduction):
- All dot products (`sdot`, `ddot`, `cdot`, `zdot`)
- All norms (`snrm2`, `dnrm2`, `scnrm2`, `dznrm2`)
- All reductions (`sasum`, `dasum`, `scasum`, `dzasum`)
- All index-finders (`isamax`, `idamax`, `icamax`, `izamax`, `isamin`, etc.)
- GEMV inner product loop (reduce over K dimension)

**V12.PRECISION** (per-stage type selection):
- `sgemm` INT8 quantized variants (matches main.cpp V12.PRECISION ablation)
- `sbgemm`, `sbgemm`, `shgemm` → BF16/F16 compute with FP32 accumulate
- `sgemv` → INT8 weights, FP32 output (matches moe_layer.cpp:115 INT8 dequant)
- Mixed precision for `dgemm` → FP32 intermediate for speed

**V12.TILE** (autotuned tile/locality):
- All Level 2 and Level 3 ops that work on matrices
- Uses SLM (P3 staged local store from Cell concepts)
- Tile size decided by cost model (H-A generated staging schedule)
- Alignment-checked by harness (H-G)

**V12.COST-GATE** (cost model selects rewrites):
- All batch operations (`gemm_batch`, `gemm_batch_strided`)
- trsm panel size decision
- Precision selection (INT8 vs FP32 vs BF16)
- Thread count → mapped to `openblas_set_num_threads` → OpenVINO thread hints
- Tile size selection for each shape (H-F local budget)

### Debug-hypothesis: grouping and pipeline design (Cell concepts)

**Grouping** (how Cell concepts help organize OpenBLAS functions):

1. **Stage 1 — Memory staging (P3, H-A):** OpenBLAS `set_matrix`/`get_matrix`
   and `omatcopy`/`imatcopy` functions → GPU uses a **ring of SLM slots**
   (depth 1–3 from manifest sweep). The contract (workload_contract.hpp:67)
   already declares `workgroup_shape`. Generator emits double/triple-buffered
   load from host-visible memory → SLM, overlapping with compute on previous
   stage.

2. **Stage 2 — Compute staging (P15):** GEMM tile loop: load A-tile into SLM,
   load B-tile into SLM, compute C-tile, write back. Each tile is a "stage"
   in the pipeline. V12.COST-GATE decides tile size; Cell concepts H-A
   generates the staging schedule from the contract.

3. **Stage 3 — Batch grouping (P16):** OpenBLAS `gemm_batch` and
   `gemm_batch_strided` → group multiple small GEMMs into one large dispatch.
   This is V12.FUSE at the dispatch level (not kernel level). Placement-aware
   dispatch order (H-C) tests linear vs swizzled vs blocked batching.

4. **Stage 4 — Precision pipelines (P19):** Each function family gets precision
   variants. V12.COST-GATE picks: INT8 for inference (tolerance 1e-2 per
   workload_contract.hpp:63), FP32 for training, BF16 for mixed precision.
   Code foot-print budget (H-H) limits shader variants.

**Pipeline design (per Cell concepts §5):**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Host memory  │ →  │ SLM staging  │ →  │ Compute EU  │ →  │ Output SLM  │ →  │ Host memory  │
│ (OpenBLAS)   │    │ (H-A ring)  │    │             │    │ (H-A ring)  │    │ (OpenBLAS)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
   ↑                 ↑                    ↑                   ↑                    ↑
   │ H-G alignment   │ H-F declared       │ V12.TILE          │ P19 code           │ V12.PRECISION
   │ rejected          budget             │ workgroup size    │ footprint budget   │ per-stage types
```

Each OpenBLAS function family maps to one or more stages. The V12 methods
determine which stages are fused (V12.FUSE), which work is elided (V12.ELIDE),
which reductions run in subgroup (V12.REDUCE), which precision is used
(V12.PRECISION), and which tile sizes are tested (V12.TILE). The cost model
(V12.COST-GATE) selects the pipeline.

---

## What we do NOT port (and why)

| Category | Reason |
|---|---|
| `xerbla` error handler | Host-side scalar; stays on CPU |
| `openblas_set_xerbla` | Same; error reporting is host-side |
| `goto_set_num_threads` / `openblas_set_num_threads_local` | Replaced by OpenVINO thread hints (§5) |
| `openblas_cancel_token` / `openblas_cancel` | Cooperative cancellation; not applicable to GPU dispatches (H-D progress word only for host polling) |
| `openblas_setaffinity` on GPU | No CPU affinity concept; H-C placement sweep replaces |
| `openblas_set_threads_callback_function` | Custom threading model; GPU has implicit parallelism via workgroups |
| `cblas_xerbla` | Error reporting; host-side |
| All `*_bf16`/`*bf16` conversion functions | Handled by OpenVINO dtype conversion; not separate kernels |
| WASM SIMD kernels (`intrin_wasm.h`) | Not on x86-64 GPU target |
| ARM/NEON kernels | Not on x86-64 target |
| POWER/AltiVec kernels | Not on x86-64 target |

---

## Verification matrix (r1–r14)

| ID | Test | Baseline function | Method tag |
|---|---|---|---|
| r1 | Elementwise ops correctness (max err 1e-5) | `saxpy`, `sscal`, `scopy`, `srot` | V12.FUSE |
| r2 | GEMV correctness (max err 1e-4) | `sgemv`, `dgemv` | V12.TILE, V12.REDUCE |
| r3 | Complex GEMV correctness (max err 1e-3) | `cgemv`, `zgemv` | V12.TILE, V12.REDUCE |
| r4 | GEMV banded correctness (max err 1e-5) | `sgbmv`, `ssbmv` | V12.ELIDE |
| r5 | Symmetric vector correctness (max err 1e-5) | `ssymv` | V12.ELIDE, V12.TILE |
| r6 | Triangular vector correctness (max err 1e-5) | `strmv`, `strsv` | V12.ELIDE, V12.TILE, V12.REDUCE |
| r7 | Symmetric matrix correctness (max err 1e-5) | `ssymm`, `ssyrk` | V12.ELIDE, V12.TILE |
| r8 | Symmetric packed correctness (max err 1e-4) | `spmv`, `sspr` | V12.ELIDE, V12.TILE |
| r9 | Triangular packed correctness (max err 1e-4) | `tpmv`, `tpsv` | V12.ELIDE, V12.TILE |
| r10 | Triangular matrix correctness (max err 1e-5) | `strmm`, `strsm` | V12.ELIDE, V12.TILE |
| r11 | TRSM panel solve correctness (max err 1e-4) | `strsm` | V12.COST-GATE |
| r12 | Memory layout transpose correctness | `somatcopy`, `imatcopy`, `geadd` | V12.FUSE, V12.TILE |
| r13 | Bulk transfer integrity (bit-exact) | `set_matrix`, `get_matrix` | V12.TILE (H-A staging) |
| r14 | Dispatch order spread (GB/s variance) | `sgemv` (large) | V12.COST-GATE (H-C) |

---

## Clean-room provenance

| Component | Source class | Notes |
|---|---|---|
| OpenBLAS API signatures | Class B (consume-only) | `cblas.h`, `common.h` headers — API surface only |
| OpenBLAS kernel algorithms | Class A (papers) | BLAS reference: Lawson et al. (1979), Dongarra et al. (1979) |
| V12 methods (FUSE/ELIDE/etc.) | Class A (project methodology) | VPU_CLEANROOM_POLICY_v0.1.md, VPU_CELL_CONCEPTS_v0.1.md |
| Cell staging/SLM concepts | Class A (papers) | arXiv:0710.2442, §2 local store; arXiv:0910.2324, §3; Intel Xe doc |
| SIMD instruction set knowledge | Class A (IntelSDM) | Intel® 64 and IA-32 Architectures Software Developer's Manual |
| OpenVINO API usage | Class B (vendor SDK) | openvino_backend.cpp:3 openvino.hpp include |

**This document is clean-room.** All V12 mappings are derived from the project's
own methodology papers and the Cell architecture papers in the source register
(VPU_CELL_CONCEPTS_v0.1.md §10). OpenBLAS function signatures are used as API
reference only (Class B). No OpenBLAS kernel source is copied or transliterated.

---

## Heterogeneous target distribution (CPU + GPU + NPU via OpenVINO)

The Ultra 5 135H has three compute devices. The V12 cost model (V12.COST-GATE)
decides per-function which device to use. OpenVINO's device priority
(`AUTO`, `MULTI`, `HETERO`) handles the routing. The distribution matrix below
uses Cell concepts for cross-device staging (H-A generated ring, H-B timeline
tags).

### Device capability summary

| Device | Strengths | Weaknesses | Best fit |
|---|---|---|---|
| CPU (P-core AVX2+VNNI) | Scalar sequential, high single-thread perf, large caches | Limited parallel lanes (8 P-cores) | Small kernels, scalar setup, trsm forward sub |
| GPU (Arc Xe-LPG 128 EU) | 128 EUs × SIMD16, high throughput, SLM | Dispatch overhead ~4–8µs, limited FP64 | Large GEMM, batched elementwise, large GEMV |
| NPU (Gaudi-inspired) | Low-power, optimized for int8/fused ops | Very limited memory (8 GB shared), no FP64 | INT8 matmul, simple activations, fixed shapes |

### OpenBLAS function → device routing

| OpenBLAS fn family | Primary device | Secondary (tandem) | V12 routing logic | Rationale |
|---|---|---|---|---|
| Level 1 elementwise (saxpy, sscal, scopy, sswap, srot) | CPU | GPU (batched) | If `N < 8192` → CPU; if batched `N ≥ 1024` → GPU via V12.FUSE | Dispatch overhead > compute for small N on GPU |
| Level 1 reductions (sdot, snrm2, sasum, isamax) | CPU | GPU (large N) | If `N > 65536` → GPU; subgroup V12.REDUCE | CPU has faster horizontal reduction for small N |
| Level 2 GEMV (sgemv, dgemv) | GPU | CPU (small) | If `M×K > 256×256` → GPU; else CPU | GPU SLM tiling wins at high arithmetic intensity |
| Level 2 banded (sgbmv, ssbmv) | GPU | — | SLM staging of band rows; V12.ELIDE | Banded matrices are regular → GPU-friendly |
| Level 2 triangular solve (strsv, trsv) | CPU | — | Sequential dependency chain | Forward sub is inherently sequential; CPU branch prediction |
| Level 3 GEMM (sgemm, dgemm) | GPU | NPU (INT8) | Large → GPU; INT8 inference → NPU via V12.PRECISION | GPU best for FP32, NPU best for INT8 at fixed shapes |
| Level 3 symmetric (ssyrk, ssymm) | GPU | — | V12.ELIDE halvs the work → GPU tile | GPU benefits from halved compute + SLM tiling |
| Level 3 triangular (strmm, trsm) | GPU (trmm) / CPU (trsm) | — | TRMM → GPU tile; TRSM → CPU forward sub | TRSM data dependency → CPU better for small panels |
| Memory layout (omatcopy, imatcopy) | GPU | — | SLM transpose tile + H-A staging | GPU SLM ideal for tiled transpose |
| Thread control (set_num_threads, get_config) | CPU | — | Always host-side | N/A on accelerator |
| Batched GEMM (gemm_batch, strided) | GPU | NPU | V12.FUSE groups → single dispatch | GPU excels at batched; V12.COST-GATE picks batch size |

### Cross-device pipeline example (GEMM → activation → gemv, as in MoE layer)

```
CPU (dispatch) → [H-A staging ring] → GPU (GEMM tile) → [H-A staging ring]
     ↓
→ NPU (INT8 matmul) → [H-B timeline sync] → GPU (V12.REDUCE) → CPU (output)
```

1. **CPU** dispatches the V12.COST-GATE decision: which ops go to which device
2. **H-A staging ring** (P15, generated from workload_contract.hpp) moves data
   - Host → GPU SLM staging slots
   - Host → NPU shared memory
3. **GPU** runs tiled GEMM (V12.TILE + V12.FUSE with bias)
4. **NPU** runs INT8 matmul for second projection (V12.PRECISION, if tolerance
   permits — matches main.cpp V12.PRECISION ablation at 1e-2 tolerance)
5. **GPU** runs subgroup REDUCE for output accumulation (V12.REDUCE)
6. **H-B timeline tags** order the transfers (one timeline semaphore per stream)
7. **H-D progress word** polls NPU completion from CPU

### Debug-hypothesis block: heterogeneous routing

**Hypothesis H-GPU-NPU-1:** Splitting a MoE FFN into `GEMM(fp32)→GPU` and
`GEMM(int8)→NPU` is slower than `GEMM→GPU` alone because the host↔NPU transfer
(~0.5ms @ 8GB shared) exceeds the compute savings for small shapes.

*Test:* Compare `gemm_batch(GEMM_fp32→GPU, GEMM_int8→NPU, GEMM_fp32→GPU)` vs
`gemm_batch(all→GPU)` across shapes 512×512 to 4096×4096.

*Kill:* if transfer overhead > 20% of GPU-only path for any shape.

**Hypothesis H-PRIO-1:** OpenVINO `AUTO:GPU,NPU,CPU` ordering wastes NPU on
shapes where GPU has spare capacity; explicit H-C placement sweep beats it.

*Test:* H-C self-test (VPU_CELL_CONCEPTS_v0.1.md §8.2) — dispatch order sweep
across GPU, NPU, CPU with linear/swizzled/blocked ordering.

*Kill:* if spread across orders < 5% (then drop P16 for this device).

**Hypothesis H-STAGE-1:** H-A staging ring depth 2 beats depth 3 for GEMM
because GPU compute throughput > SLM bandwidth, so one-slot lookahead saturates.

*Test:* H-A generated staging at depths 1, 2, 3 on large GEMM (8192×8192),
measured against hand-tuned OpenBLAS CPU reference.

*Kill:* if depth-2 is >5% slower than depth-3 at large shapes.

### What we distribute vs tandem

**Distributed (independent work):**
- Batch dimension split across GPU + NPU (each device handles different batch items)
- `gemm_batch`: OpenVINO `MULTI:GPU,NPU` splits batch groups
- Elementwise ops: GPU handles large vectors, CPU handles small ones
- NPU takes fixed-shape INT8 kernels (its compiler works best with known shapes)

**Tandem (pipeline stages):**
- MoE pipeline: GPU GEMM → NPU activation → GPU second GEMM
- This needs H-A staging (overlap transfer with compute) + H-B timeline tags
- Currently **deferred** (H-E bounded claim, per VPU_CELL_CONCEPTS §5/H5) —
  stays in suite as candidate, not default (R9: ablations not bundling)

### Device query mapping (VPU_CELL_CONCEPTS §2)

| OpenBLAS concept | VPU Cell concept | OpenVINO API | meta.json field |
|---|---|---|---|
| `openblas_get_num_procs` | Query unit count at runtime (§2, last row) | `ov::Core::get_available_devices` + `OPTIMAL_NUMBER_OF_THREADS` | `hardware.cpu` |
| Thread affinity | Placement-sensitive bandwidth (§2) | H-C dispatch order sweep (§5.3) | `placement_sweep` |
| Local buffer size | 256 KB local store per SPE (§2) | `maxSharedMemorySize` per Xe-core (query at runtime) | `slm_bytes_limit` |
| DMA queues | 16 outstanding DMAs per SPE (§2) | Fence pool vs timeline semaphore (H-B) | `queue_depth` |
| Inter-CPU bandwidth | EIB bandwidth (§2) | Cross-device transfer bandwidth | `transfer_gbps` |

### SIMD / compute capability detection

The host-side OpenBLAS baseline queries CPU SIMD at runtime via `cpuid_x86.c`.
For the GPU path, we need:

| Feature | Intel Ultra 5 135H | GPU (Xe-LPG) | NPU | OpenVINO mechanism |
|---|---|---|---|---|
| AVX2 | Yes (P-core) | No | No | `ov::device::capabilities` → CPU target |
| AVX-VNNI | Yes (Meteor Lake) | No | Yes | INT8 dot product on NPU |
| FP64 | Yes (scalar) | 1/16 throughput | No | Avoid FP64 on GPU/NPU |
| BF16 | P-core: no; NPU: yes | Yes (Xe-LPG) | Yes | V12.PRECISION for bf16 |
| INT8 | P-core VNNI; E-core no | Limited | Yes (primary) | V12.PRECISION for i8 |

**We do NOT need** (wrong target or not present):
- AVX-512 on CPU — Meteor Lake mobile SKU does not have it
- AMX — server-only
- WASM SIMD — not targeting browser
- ARM NEON — not x86-64
- GPU-specific ISA assembly — OpenVINO compiler handles this

**We DO need** (present on target):
- SSE2 as fallback baseline (all x86-64 must support it)
- AVX2 + FMA3 as primary CPU SIMD (Raptor Cove P-cores)
- AVX-VNNI for INT8 dot products on CPU (if porting OpenBLAS INT8 kernels)
- OpenVINO GPU plugin for Arc Xe-LPG
- OpenVINO NPU plugin for Intel NPU (Gaudi-style)

### What gets ADDED, not removed

Per user: *"let us just add stuff than remove"*

The following OpenBLAS functions **gain GPU/NPU capability** without removing
their CPU path:

| Function | CPU path (existing) | GPU path (added) | NPU path (added) |
|---|---|---|---|
| `sgemm` | AVX2/FMA3 kernel | OpenVINO MatMul (tiled via V12.TILE) | OpenVINO INT8 MatMul (V12.PRECISION) |
| `sgemv` | AVX2 kernel | OpenCL kernel (SLM tile, V12.TILE + V12.REDUCE) | NPU matvec (fixed shapes) |
| `saxpy` | SSE2 kernel | OpenVINO Eltwise (V12.FUSE with consumer) | N/A (too scalar) |
| `sdot` | AVX2 horiz sum | Subgroup reduce (V12.REDUCE) | NPU dotprod (VNNI equivalent) |
| `ssymv` | AVX2 + symmetry elision | SLM tile + upper-triangle skip (V12.ELIDE) | N/A |
| `strsm` | Blocked CPU algo | Panel solve on GPU, CPU fallback for small panels | N/A |
| `sgemm_batch` | OpenMP threading | V12.FUSE into single dispatch | MULTI device split |
| `sbgemm` | (new) | OpenVINO BF16 MatMul | NPU BF16 (native) |

---

## AVX-512 and AMX equivalents on GPU (and why GPU is faster anyway)

> **Question:** Do AVX-512 and AMX have GPU equivalents? Can GPU execute these
> faster than CPU?

**Short answer:** Yes — the GPU doesn't need 1:1 instruction equivalents.
The throughput comes from parallelism, not wider registers.

### CPU SIMD landscape (Intel Core Ultra 5 135H)

| SIMD | Lane width | Elements/insn | On Ultra 5 135H? | GPU equivalent |
|---|---|---|---|---|
| SSE2 | 128-bit | 4× f32 | Yes (baseline) | N/A (scalar fallback) |
| AVX2 | 256-bit | 8× f32, 32× i8 | Yes (P-core) | 16 workitems × 1 f32 each = 16× f32 |
| AVX-512 | 512-bit | 16× f32, 64× i8 | **No** (mobile SKU) | 16 subgroup lanes × 1 op each |
| AMX | 2D tile | 16×16 i32 from i8 | **No** (server only) | SLM tile + subgroup MAC chain |
| VNNI (AVX-VNNI) | 256-bit | 4× i8→i32 dot | Yes (P-core) | GPU integer MAC pipeline |

### GPU equivalence (Intel Arc Xe-LPG, 128 EU)

**AVX-512 → GPU subgroup:** A single Xe-LPG subgroup has 16 lanes. Each lane
holds one element. 16 lanes × 1 FMA = 16-element-wide reduction in one
instruction — this is the GPU's "AVX-512." But Xe-LPG has 128 EUs × 8
subgroups = 1024 concurrent subgroups. So one GPU dispatch processes **1024×
more elements** than one AVX-512 instruction in the same cycle.

**AMX → GPU SLM tile + MAC chain:** AMX tiles a 16×16 i8→i32 matrix multiply
into one chip. Xe-LPG does the same via SLM: load 16×16 tiles of A and B into
shared local memory, then each subgroup computes a 16×16 output tile via
4-element MAC chains. The difference is scale: GPU has 128 EU copies of this
pattern running in parallel.

**VNNI → GPU integer dot:** Xe-LPG OpenCL supports 4×int8 → int32 dot product
via `intel_sub_group_accumulate` or raw SIMD multiply + accumulate. Same
operation, but 1024 subgroups vs 1 P-core.

**H-F tile budget (debug-hypothesis):**
OpenBLAS AMX kernels use 16×16 tiles (matches AMX TMUL). Xe-LPG SLM is
queryable (§2). Hypothesis: SLM tile of 16×16×4 B = 1 KiB per tile pair ×
128 EUs = 128 KiB, fits in 192 KiB SLM (typical Xe-LPG). If exceeds,
fall back to 8×8 tiles (V12.COST-GATE).

### Why GPU > CPU even without matching SIMD

| Factor | CPU (8 P-cores) | GPU (128 EU) | GPU advantage |
|---|---|---|---|
| Execution width | 8 f32 (AVX2) | 16 per subgroup × 128 EUs | 256× parallelism |
| Memory bandwidth | ~77 GB/s (shared L3) | ~200 GB/s (system memory via GPU driver) | 2.6× |
| Dispatch overhead | <1µs | ~4–8µs | CPU wins small |
| Branch penalty | mispredict = 10–20 cycles | predication, no penalty | GPU wins on irregular |
| Frequency | 4.8 GHz | 1.3–1.6 GHz | CPU wins per-thread |
| Active threads | 8 | 1024+ | GPU wins throughput |

**Debug-hypothesis H-SIMD-GPGPU-1:** GPU beats CPU for:
- GEMM: M×N×K > 256×256×256 (enough work to hide dispatch + saturate 128 EU)
- GEMV: M > 512, K > 256 (enough rows to keep 128 EU busy)
- DOT/Norm: N > 65536 (need 1024 subgroups to amortize)
- Elementwise: N > 8192 (enough elements to fill 1024 lanes)

Kill condition: if GPU speedup < 1.5× at the threshold shape after 20+
measured iterations with 95% CI excluding 1.0 (R1), fall back to CPU.

### SIMD mapping table (what we compile vs what we use)

| OpenBLAS SIMD flag | Compile-time? | Runtime check | GPU mapping |
|---|---|---|---|
| `HAVE_SSE2` | Required (x86-64 baseline) | Always available | CPU fallback baseline |
| `HAVE_SSE3`, `HAVE_SSE4_1` | Optional | CPUID check | CPU only — used for ssSE4 integer ops |
| `HAVE_AVX` | Optional | CPUID check | CPU only |
| `HAVE_AVX2` | Optional | CPUID check | CPU P-core; V12.COST-GATE uses it for INT8 |
| `HAVE_AVX512F/VL/BF16` | **Skip** (not on mobile) | — | No GPU equivalent needed — GPU has its own matrix path |
| `HAVE_AMXBF16` | **Skip** (not on mobile) | — | GPU SLM tile replaces AMX tile |
| `HAVE_FMA3` | Optional | CPUID check | CPU only; GPU has FMA in every EU |
| `HAVE_VNNI` | Optional (Meteor Lake) | CPUID check | GPU integer MAC pipeline |

**What we do NOT compile for the GPU path:**
- None of the AVX-512 or AMX paths — the GPU compiler (OpenVINO / OpenCL)
  generates GPU-specific SIMD from the OpenVINO IR. We write `MatMul`,
  `Eltwise`, `ReduceSum` ops; the GPU backend lowers them to GPU SIMD.
- No hand-written x86 assembly needed for GPU — that is the CPU baseline only.

