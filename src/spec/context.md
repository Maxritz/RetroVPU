# Context Management (P3 Staged Local Store)

## Source Citation
- arXiv 0910.2324, section 3 (local store and memory hierarchy)
- VPU_CELL_CONCEPTS_v0.1.md, section 2 (SLM query)

## Overview
The Context class manages device initialization and discovery.

## Device Query
- Query available OpenVINO devices
- Record memory capacity per device
- Cache device capabilities

## Local Memory (P3 Analogue)
On the L14, shared local memory maps to:
- **CPU**: Thread-local cache / L1
- **NPU**: On-chip SRAM (size queried at runtime)
- **GPU**: Shared local memory (SLM) per-Xe-core

## UNSPECIFIED-BY-PAPER
- Exact SLM size for Xe-LPG must be queried (§2, last row of Cell concepts)
- Cache hierarchy characteristics to be measured