# MoE Layer Architecture (Clean-Room Derived)

## Source Citations
- **P3 Staged Local Store**: arXiv 0910.2324, section 3 (local store management)
- **P10 Sparse Compute**: VPU_CLEANROOM_POLICY_v0.1.md, section 5 (method catalogue)
- **H-E Bounded Dynamic Claim**: VPU_CELL_CONCEPTS_v0.1.md, section 5.5
- **Cell Architecture**: arXiv 0710.2442, section 2 (core organization)

## Architecture

### Overview
The MoE layer implements a sparse Mixture-of-Experts architecture with:
- Multiple expert feed-forward networks
- Token dispatch to top-K experts based on routing scores
- Merge of expert outputs weighted by routing coefficients

### Dispatch Algorithm (P16: Placement-Aware Dispatch Order)

1. For each input token, compute routing scores to all experts
2. Select top-K experts per token
3. Group tokens by expert for efficient batching (V12.FUSE)

### Compute Algorithm (P3: Staged Local Store)

1. Load expert weights into local memory (shared memory on GPU, stack on CPU)
2. Apply activation function (GELU) on intermediate computations
3. Project output back to model dimension

### Merge Algorithm (V12.REDUCE)

1. Accumulate expert outputs weighted by routing scores
2. Use subgroup-parallel reduction for efficiency

## V12 Method Tags

| Tag | Applied When | Must Beat | Decided By |
|-----|-------------|-----------|-----------|
| V12.FUSE | Multiple token dispatches to same expert | N separate dispatches | total μs, bytes moved |
| V12.ELIDE | Skip experts with no assigned tokens | Compute all experts uniformly | μs, work skipped |
| V12.REDUCE | Merge expert outputs | Serial accumulation | μs at equal output |
| V12.PRECISION | INT8/BF16 compute vs FP32 | All-FP32 baseline within tolerance | μs + max/rms error |
| V12.TILE | Tiled matrix multiply within expert | Naive matmul | μs |

## UNSPECIFIED-BY-PAPER

- Routing function initialization (gap: papers don't specify exact init)
- Expert capacity factor (gap: Sequoia SC 2006 mentions factor but not exact value)
- Load balancing coefficient (gap: to be determined by experiment)

## Parameters

| Parameter | Paper Source | Value |
|-----------|-------------|-------|
| Activation | GELU from transformer papers | GELU |
| Init | Xavier/Glorot | arXiv: 1003.0486 section 4.2 |
| Tokens per expert | Expert capacity factor | 1.25x mean (experiment) |