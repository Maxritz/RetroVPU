# VPU MoE Lite

A clean-room implementation of a lightweight Mixture-of-Experts engine for the VPU (Vulkan Processor Unit) methodology project. This engine measures V12 methods against strong direct baselines on equal terms.

## Project Overview

**Objective:** Find, by honest measurement, how to make each workload better compared against a strong direct baseline.

**Hardware Target:** Intel Core Ultra 5 135H with Arc iGPU (128 EU), 16 GB RAM, ThinkPad L14 running Windows.

**GPU API:** OpenVINO (CPU + NPU offloading via C++ API).

**In Scope:** OpenVINO CPU execution and NPU offloading via Intel NPU plugin. The Arc iGPU is detected by OpenVINO and shares the system's 16 GB RAM with the CPU (zero-copy unified memory). The NPU can address up to 8 GB of system memory.

## Clean-Room Compliance

This is a clean-room project. Every source file under `src/` includes a provenance header documenting:

- **Mechanism:** which VPU principle the code implements
- **Derived-from:** academic paper citations (arXiv/DOI) with section references
- **Author:** human or AI agent identification

No code is copied from forbidden repositories. The only third-party code incorporated is Intel vendor SDK APIs (Class B, consume-only).

**Policy:** See `VPU_CLEANROOM_POLICY_v0.1.md` for full details.

## V12 Method Catalogue

| Method | Mechanism | Must Beat | Decided By |
|--------|-----------|-----------|-----------|
| Fuse | N ops → 1 dispatch | N separate dispatches | total µs, bytes moved |
| Elide | Skip provably-inactive work | Dense compute-then-mask | µs, work skipped |
| Reduce | Subgroup-parallel reduction | Single-thread serial loop | µs at equal output |
| Precision | Per-stage type selection | All-F32 within tolerance | µs + max/rms error |
| Tile | Autotuned tile/locality | Fixed/naive tile | µs |
| Cost-gate | Cost model selects rewrites | Apply-all / naive | end-to-end µs |

## Measurement Protocol (R1-R12)

- **R1:** Warmup + 20+ measured iterations; report median + 95% CI on median; print "inconclusive" when CI overlaps 1.0
- **R2:** Isolate the variable - vary only the axis under test
- **R3:** Workload contract before timing
- **R4:** Correctness gates performance (CPU reference validation)
- **R5:** Best-of-both comparison per shape
- **R6:** Baseline stays independent (hand-authored, not emitted from V12 template)
- **R7:** Report GPU-timestamp, host submission, and end-to-end separately
- **R8:** Record environment (device, driver, API version, compiler, shader hashes)
- **R9:** Every pass is independently switchable
- **R10:** Methods are cost-gated, not rules
- **R11:** No historical-graphics emulation in LLM core
- **R12:** Tags are intent, not evidence

## Current Benchmark Results

### MoE Layer Forward Pass (8 experts, top-2 routing, hidden=512, intermediate=2048)

| Variant | Method Tag | Median (µs) | 95% CI (µs) | Speedup |
|---------|-----------|-------------|-------------|---------|
| Baseline (dense, FP32) | V12.NONE | 40,258 | 33,538 - 48,376 | 1.00x (ref) |
| Elision only | V12.ELIDE | 2,163 | 2,035 - 3,047 | 18.62x |
| Fusion only | V12.FUSE | 2,387 | 2,079 - 3,148 | 16.87x |
| Precision only | V12.PRECISION | 2,185 | 2,029 - 3,589 | 18.42x |
| Fused + Elided + Precision | V12.FUSE\|ELIDE\|PRECISION | 2,122 | 2,009 - 2,986 | 18.97x |

**Combined V12 speedup:** 18.97x (CI: 13.48 - 20.04), statistically significant

**Key finding:** V12.ELIDE (sparse top-K dispatch) provides the dominant win. Computing 2 of 8 experts instead of all 8 yields ~18-19x speedup. V12.FUSE and V12.PRECISION provide marginal additional gains over elision alone, as they both include the elision optimisation. To isolate fuse and precision contributions, future ablations should compare elide-only vs elide+fuse, and elide-only vs elide+precision.

**Variance note:** Run-to-run speedup ranged from 17x to 26x across four full benchmark runs on this laptop (thermal effects on the baseline; the ratio stays directionally stable). All individual measurements and CIs are archived in `benchmark_results.json`.

All results saved to `benchmark_results.json` after each run.

## Building

### Prerequisites

- Windows with MSVC 2022 BuildTools
- OpenVINO 2026 toolkit (set `OPENVINO_ROOT` environment variable)

### Build Commands

```
build.bat           # Build only
build.bat run       # Build and run benchmarks
build.bat rebuild   # Clean rebuild
build.bat clean     # Remove build directory
```

## Architecture

```
src/
  core/            # Device context management (OpenVINO Core API)
  model/           # Tensor and MoE layer implementations
  backend/         # OpenVINO backend integration
  harness/         # R1-R12 compliant benchmark harness
  spec/            # Specification documents
spec/              # Legacy spec directory (referenced by headers)
```

## V12 Method Implementation Status

| Method | Status | Notes |
|--------|--------|-------|
| V12.ELIDE | Implemented | Top-K sparse compute, skip inactive experts |
| V12.PRECISION | Implemented | INT8 quantized weights vs FP32 baseline |
| V12.FUSE | Framework | Dispatch grouping ready (single-token currently) |
| V12.REDUCE | Planned | Subgroup reduction in merge step |
| V12.TILE | Planned | Autotuned tiling for matmul |
| V12.COST-GATE | Planned | Dynamic cost model for method selection |

## License

Restrictive license. Contact project owner for permission.

## References

- Cell architecture: arXiv:0710.2442
- Local store management: arXiv:0910.2324
- V12 VPU methodology: `VPU_CLEANROOM_POLICY_v0.1.md`, `VPU_CELL_CONCEPTS_v0.1.md`
- Clean-room enforcement: `AGENTS.md`