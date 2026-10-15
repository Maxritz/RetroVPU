# Claude Code Instructions for VPU Project

## Quick Start

Before any `src/` work, confirm with user:
1. Hardware target (CPU, GPU, RAM, benchmark machine)
2. OS target (Windows, Linux, or both)
3. Language target (host + kernel languages)
4. GPU API targets (Vulkan, OpenVINO, HIP, CUDA — priority order)
5. Implementers (humans, AI agents, or both)
6. Oracle policy (default: no forbidden black-box usage)
7. Output license

## Clean-Room Rules (MUST FOLLOW)

### What You Can Read
- Papers (arXiv, DOI) — cite by section
- Vendor API specs for confirmed APIs only (OpenVINO documentation)
- Standards (IEEE 754)
- This project's own files

### Web Access Permissions
- arxiv.org — always allowed (academic research)
- OpenVINO documentation — allowed (confirmed API target)
- GitHub — only with formal Class D approval logged in exposure ledger

### What You MUST NOT Read
- **NEVER** read code from: llama.cpp, ggml, vllm, sglang, TensorRT-LLM, mlc-llm, exllamav2, ollama, text-generation-inference, mlx-lm, candle
- **NEVER** seek implementation details from LLM system repos
- **NEVER** reproduce identifiers: `ggml_`, `GGML_`, `llama_`, `LLAMA_`
- **NEVER** create files matching: `**/ggml*`, `**/llama.cpp*`

### For Every `src/` File
Add this header:
```text
// VPU-CR-PROVENANCE
// Mechanism:  <principle id>
// Derived-from: <arXiv/DOI id>, <section>
// Spec entry: spec/<file>#<anchor>
// Author:     agent:claude@<session>
// Exposure:   none
```

### For Every `src/` Commit
Add these trailers:
```
Clean-Room-Source: <arXiv/DOI id>, <section>
Clean-Room-Exposure: none
```

## Measurement Protocol (R1-R12)

**Before any performance claim:**
- R1: Warm-up + repeated batches, report median + spread + 95% CI
- R2: Isolate variables — vary only the axis under test
- R3: Write a workload contract first (op, M/N/K, dtype, layout, dispatch count)
- R4: Validate correctness against CPU reference before timing
- R5: Compare best-V12 vs best-direct, per shape
- R6: Baselines are hand-authored, not from V12 templates
- R7: Report GPU execution, host submission, end-to-end separately
- R8: Record GPU, driver, Vulkan version, workgroup shape, shader hashes
- R9: Each method independently switchable (no bundling)
- R10: Cost-gate decisions, don't apply blindly
- R12: Tags like `V12.FUSE`, `V12.ELIDE` are intent, not evidence

## Cell Concepts (Adopted from Papers Only)

- **P3**: Staged local store — shared memory as GPU analogue
- **P15** (proposed): Generated staging schedules
- **P16** (proposed): Placement-aware dispatch order
- **P17** (proposed): Progress-word signalling (gated on litmus test)
- **P18** (proposed): Bounded dynamic claim per dispatch
- **P19** (proposed): Declared local and code budgets

**Drop:** SPE-to-SPE DMA, dual-issue hand scheduling

## If You Encounter Class C Content

1. **STOP** — do not copy or reproduce
2. Record exposure in the exposure ledger
3. Move affected modules to `quarantine/`
4. Re-derive from Class A spec in new session

**Important:** I cannot access, read, or derive implementation from forbidden repositories even if you suggest them as reference. If you point to a forbidden source, I must refuse and suggest finding academic papers instead. Override requires formal Class D approval.

## Verification Checklist (Before Commit)

- [ ] Provenance header present with citation
- [ ] Commit trailers present (`Clean-Room-Source`, `Clean-Room-Exposure: none`)
- [ ] No forbidden identifiers or paths
- [ ] At least one citation to a Class A paper per function
- [ ] Workload contract written before timing code
- [ ] Results include 95% CI and environment metadata