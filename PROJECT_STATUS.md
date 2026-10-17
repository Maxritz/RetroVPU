# V12 GEMV Implementation Status

## Mission Status: IN PROGRESS

### Current State:
- **Level 1 BLAS ops** (SDOT, SNRM2, SAXPY, SASUM, SSCAL, ISAMAX): Working on GPU via OpenVINO
- **Level 2 BLAS ops** (SGEMV, DGER, DGER): Implementation complete, GPU execution testing in progress
- **Level 3 BLAS ops**: Not yet implemented

### Recent Results (Level 1 GPU Execution):
```
sdot    [N=16384] V12=sgemv-l1-sdot  vs  CPU-Ref  speedup=3.6x  [SIGNIFICANT]
snrm2   [N=16384] V12=snrm2-v12      vs  CPU-Ref  speedup=1.0x  [~equal]
saxpy   [N=16384] V12=saxpy-v12      vs  CPU-Ref  speedup=1.2x  [INCONCLUSIVE]
sasum   [N=16384] V12=sasum-v12      vs  CPU-Ref  speedup=4.1x  [SIGNIFICANT]
sscal   [N=16384] V12=sscal-v12      vs  CPU-Ref  speedup=1.3x  [INCONCLUSIVE]
isamax  [N=32768] V12=isamax-v12     vs  CPU-Ref  speedup=0.4x  [INCONCLUSIVE]
```

### Implementation Features

#### 1. Optimized Baseline (AVX-512 SIMD)
```cpp
// AVX-512 SIMD with 8-wide accumulation
__m512 sum = _mm512_setzero_ps();
for (size_t kk = 0; kk < N; ++kk) {
    __m512 x_val = _mm512_set1_ps(x[kk * incx]);
    __m512 a_val = _mm512_loadu_ps(A + a_row * lda + ii);
    sum = _mm512_fmadd_ps(x_val, a_val, sum);
}
```

**Performance Optimizations:**
- AVX-512 instructions with FMA operations
- Cache-blocking: 64×64 tiles for FP32, 32×32 for FP64
- Multi-threading with hardware-concurrency detection
- Software prefetching for memory bandwidth
- 8-wide loop unrolling for instruction-level parallelism

#### 2. V12 GPU Backend (OpenVINO)
Implemented actual GPU kernel execution using OpenVINO API:
- RunDot: MatMul-based reduction (V12.REDUCE)
- RunGemv: MatMul-based GEMV (V12.TILE + V12.FUSE)
- RunGer: Outer product via MatMul (V12.FUSE)
- RunBatchGemv: Batched MatMul (V12.FUSE + V12.TILE)
- RunScal: Element-wise multiply (V12.FUSE)
- RunAxpy: Fused multiply-add (V12.FUSE)
- RunAmax: Reduction-based max index (V12.REDUCE)

#### 3. Enhanced V12 GEMV Routing
Smart dispatch routing:
- Large matrices (M*N >= 512K elements): Optimized baseline (AVX-512 + cache-blocking)
- Small matrices: V12.GPU implementation via OpenVINO

### R1-R12 Compliance

**R1: P0 Protocol** - Warmup + repeated batches, median + spread, 95% CI
**R2: Variable Isolation** - Same shape, bytes, dispatch for fair testing
**R3: Workload Contract** - Machine-readable workload description
**R4: Correctness Gates** - CPU reference validation before timing
**R5: Best-of-Both** - V12 vs optimized baseline per shape
**R6: Baseline Independence** - Hand-authored optimized baseline
**R7: Timing Boundaries** - GPU-timestamp, host, end-to-end separated
**R8: Environment Recording** - Device, driver, API version recorded
**R9: Ablations** - Independent toggleable methods
**R10: Cost-Gated Methods** - Cost model selects rewrites
**R11: Clean-Room Implementation** - No historical-graphics emulation
**R12: Tag Validation** - Tags label intent, ablations validate

### Build & Run Instructions

**Prerequisites:**
```bash
# Windows with MSVC 2022 BuildTools
# OpenVINO 2026+ toolkit (OPENVINO_ROOT)
# Intel Arc iGPU (Xe-LPG) or compatible
# Compiler supporting AVX-512
```

**Build Commands:**
```powershell
$env:OPENVINO_ROOT="C:\Users\rina0423\Desktop\openvino_toolkit"
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target benchmark_vpu_blas
```

**Run:**
```cmd
cd build\src\Release
benchmark_vpu_blas.exe
```

### Test Results
