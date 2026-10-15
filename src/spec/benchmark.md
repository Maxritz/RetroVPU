# Benchmark Methodology

## R1: No Claim from One Measurement
- Warmup: 5 iterations minimum
- Measured: 20+ iterations minimum
- Report: median, mean, std dev, min, max
- 95% confidence interval on ratio
- If CI overlaps 1.0 → **inconclusive**

## R2: Isolate the Variable
- Baseline and V12 variant share:
  - Same M/N/K dimensions
  - Same parallelism level
  - Same bytes moved
  - Same dispatch shape
  - Only the method under test varies

## R3: Contract Before Timing
- WorkloadContract must be created before any RunBenchmark call
- Contract includes: op, M/N/K, dtype, layout, tolerance, dispatch count
- Hash of source files included for traceability

## R4: Correctness Gates Performance
- Reference implementation computed on CPU
- Max error threshold: 1e-2 (configurable per test)
- RMS error threshold: 1e-2
- Every output must be written completely

## R5: Best-of-Both
- Compare best V12 variant timing vs best direct baseline
- Report speedup ratio with CI
- Never compare to weaker variant or stale run

## R6: Baseline Stays Independent
- Direct baselines hand-authored, not emitted from V12 templates
- Same tuning budget applied to both

## R7: Measure Right Boundaries
- GPU timestamp execution time measured separately
- Host submission/recording time measured separately
- End-to-end includes both

## R8: Record Environment
- GPU name and EU count
- Driver version
- API/runtime version
- Compiler version
- Workgroup/tile shape
- Shader/source hashes