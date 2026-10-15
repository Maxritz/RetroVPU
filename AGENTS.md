# VPU Methodoloy Agent Instructions (Based on VPU v0.1)

## Project Overview

This is a clean-room project building a VPU methodology for an LLM engine. The objective is to find, by honest measurement, how to make each workload better compared against a strong direct-Vulkan baseline on equal terms. 

**Hardware target (to confirm):** Intel Core Ultra 5 135H with Arc iGPU (128 EU) and 16 GB RAM on a ThinkPad L14, running Windows.

## Core Principles

### The V12 Principle
- Every method is measured against the strongest independently tuned standard counterpart
- A method is credited as a V12 win only when an isolated, statistically significant ablation shows it beats that counterpart
- Methods that don't pay are logged as candidates, not deleted
- Name, tagging, or describing a method proves nothing on its own — numbers do

### Method Catalogue
| Method | Mechanism | Must Beat | Decided By |
|--------|-----------|-----------|------------|
| Fuse | N ops → 1 dispatch | N launches at equal parallelism | total µs, bytes moved |
| Elide | Skip provably-inactive work | dense-compute-then-mask | µs, work skipped |
| Reduce | Subgroup-parallel reduction | single-thread serial loop | µs at equal output |
| Precision | Per-stage type selection | all-F32 within tolerance | µs + max/rms error |
| Tile | Autotuned tile/locality | fixed/naive tile | µs |
| Cost-gate | Cost model selects rewrites | apply-all / naive | end-to-end µs |

## Clean-Room Policy Essentials

### Source Classes
- **Class A (Allowed):** Papers (cite by arXiv/DOI + section), vendor specifications for selected APIs, standards (IEEE 754), textbooks
- **Class B (Consume-only):** Vendor runtimes and SDKs — call public APIs, do not read/copy implementation
- **Class C (Forbidden):** LLM system repos, code, docs, etc. — listed in denylist
- **Class D (Conditional):** Requires logged approval — paper companion code, public format specs, non-allowlisted web pages

### Provenance is Mandatory
Every source file under `src/` MUST have this header:
```text
// VPU-CR-PROVENANCE
// Mechanism:  <principle id, e.g. P3 staged local store>
// Derived-from: <arXiv or DOI id>, <section / equation / figure>
// Spec entry: spec/<file>#<anchor>
// Author:     human:<initials> | agent:<model id>@<session id>
// Exposure:   none
```

Every commit touching `src/` MUST have these trailers:
```text
Clean-Room-Source: <arXiv or DOI id>, <section>
Clean-Room-Exposure: none
```

## Roles and Separation

AI agents are implementers. Implementers may read:
- Specifications
- Class A papers
- Class B vendor documentation

Implementers may NOT read:
- Class C source code, docs, issues, PRs, snippets
- Any non-allowlisted web pages (arxiv.org always allowed; other domains only when their API is selected)

## Enforcement Controls

### E1: Denylist Scan
- Pre-commit and CI scan for forbidden repo names, package names, identifier prefixes, path names
- Initial seeds: llama.cpp, ggml, vllm, sglang, TensorRT-LLM, mlc-llm, exllamav2, ollama, text-generation-inference, mlx-lm, candle
- Forbidden identifier prefixes: ggml_, GGML_, llama_, LLAMA_
- Forbidden path patterns: **/ggml*, **/llama.cpp*

### E2: Fingerprint Overlap
- CI checks normalized token k-grams (k=10, winnow window=4) against Class C snapshot
- **Fail rule:** Any shared fingerprint covering 12+ normalized tokens fails CI
- Cannot catch: heavy reordering or re-expression

### E3: Structural Hash
- CI checks normalized function-level syntax hashes (identifiers/literals replaced)
- **Fail rule:** Matching hash on function of 8+ statements fails CI
- Cannot catch: structures rewritten by hand with different control flow

### E4: Constants and Tables
- CI checks numeric arrays of 16+ entries, codebooks, lookup tables, tuning constants
- **Fail rule:** Any match fails CI
- Constants that appear in Class A paper are exempt (must cite paper)

### E5: Provenance Gate
- Commit hook checks `src/` commits have the two required trailers
- `Clean-Room-Exposure: none` required for `src/` commits
- Exposed work goes to `quarantine/`

### E6: Exposure Ledger
- Append-only ledger, reviewed
- Fields: `date, actor, actor_type, item_viewed, class, affected_modules, action, reviewer`
- Any exposure taints affected modules until rewritten from spec

### E7: Dependency Gate
- No vendored source code except SDKs listed in Class B
- New dependency requires Class A or Class B entry + ledger line
- Software bill of materials regenerated on each release

### E10: Monthly Audits
- Full scan with E1-E4 against current tree
- Random sampling of 3 modules: check headers, verify spec citations
- Second implementer re-derives one function from spec in new session
- Failure to re-derive means spec is incomplete — fix the spec

## AI Agent Specific Controls

1. **Instructions:** Prohibit reproducing Class C code; require Class A citation for every function
2. **Workspace:** No Class C repo checkouts or files in workspace
3. **Network:** Web access limited to allowlist (arxiv.org always allowed; OpenVINO documentation and GitHub (for clean-room research only under Class D approval required); vendor API documentation for confirmed API targets)
4. **Output gate:** Every agent-written file passes E1-E4 before commit
5. **Records:** Agent model ID and session ID in provenance header
6. **Limitation:** Agent memory cannot be inspected — E2-E4 are the backstop against memorized code

### Exposure Consequences

**I cannot access, read, or derive implementation from forbidden (Class C) repositories**, even if the user suggests it as a reference. If a user points to a forbidden source:

1. **Do NOT** open, read, or browse the repository
2. **Notify** the user that this violates clean-room policy
3. **Suggest** finding academic papers instead (arXiv.org is always allowed)
4. **If accidental exposure occurs**: 
   - Record immediately in the exposure ledger
   - Move affected modules to `quarantine/`
   - Re-derive from spec in a new session

**Note:** The user may override this restriction, but doing so requires formal Class D approval logged in the exposure ledger with: date, approver, scope, and reason.

### Exposure Consequences

**I cannot access, read, or derive implementation from forbidden (Class C) repositories**, even if the user suggests it as a reference. If a user points to a forbidden source:

1. **Do NOT** open, read, or browse the repository
2. **Notify** the user that this violates clean-room policy
3. **Suggest** finding academic papers instead (arXiv.org is always allowed)
4. **If accidental exposure occurs**: 
   - Record immediately in the exposure ledger
   - Move affected modules to `quarantine/`
   - Re-derive from spec in a new session
### Overrides 

- If the user asks for a specific git to be read and researched for ideas, it is allowed
**Note:** The user may override this restriction, but doing so requires formal Class D approval logged in the exposure ledger with: date, approver, scope, and reason.

## Violation Response

| Level | Trigger | Action |
|-------|---------|--------|
| L1 Review | E2/E3 hit below threshold | Human reviewer decides; record decision + reasoning in ledger |
| L2 Quarantine | E1 hit, or E2/E3 at/above threshold, or E4 match | Move module to `quarantine/`; re-derive from spec in new session |
| L3 Confirmed | Verbatim/identifier leak, confirmed exposure, false provenance | Quarantine, re-derive, notify user; decide history rewrite policy |

**Important:** Renaming is not a fix. Re-derive from spec.

## Cell Concepts Integration (from VPU_CELL_CONCEPTS_v0.1)

### Adopted Concepts
- Software-managed local store (P3) — shared memory on GPU is analogue
- Asynchronous DMA with overlap — generated from contract (H-A)
- DMA alignment rules — harness check (H-G)
- DMA tags/fences — timeline values (H-B)
- Local budgets with guard arenas (H-F)
- Code footprint budget (H-H)

### Dropped Concepts
- SPE-to-SPE DMA (GPU workgroups can't read each other's shared memory)
- Dual-issue pipelines and hand scheduling (GPU compiler handles)

### Principle Candidates (P15-P19, Proposed Status)
- P15: Generated staging schedules with timeline-value slot waits
- P16: Placement-aware dispatch order
- P17: Progress-word signalling (gated on litmus test)
- P18: Bounded dynamic claim per dispatch
- P19: Declared local and code budgets with guard arenas

## Intake Questions (Must Ask Before `src/` Work)

1. **Hardware target** — CPU, GPU, memory type/size, benchmark machine
2. **OS target** — Windows, Linux, or both
3. **Language target** — Host and kernel languages
4. **GPU API targets** — Vulkan, OpenVINO, ROCm/HIP, CUDA, other (priority order)
5. **Implementers** — Humans, AI agents, or both
6. **Oracle policy** — Allowed to run forbidden systems as black-box? (default: no)
7. **Output license** — Which license?

## Measurement Protocol (R1-R12)

- **R1:** No claim from one measurement — warm-up + repeated batches, median + spread, 95% CI on ratio
- **R2:** Isolate the variable — vary only the axis under test
- **R3:** Contract before timing — machine-readable workload contract with op, M/N/K, dtype, layout, etc.
- **R4:** Correctness gates performance — validate against CPU reference within tolerance
- **R5:** Best-of-both — compare best-V12 vs best-direct per shape
- **R6:** Baseline stays independent — direct baselines are hand-authored, not emitted from V12 template
- **R7:** Measure right boundaries — GPU-timestamp execution, host submission, end-to-end separately
- **R8:** Record environment — GPU, driver, Vulkan version, compiler, workgroup shape, shader hashes
- **R9:** Ablations, not bundling — every pass independently switchable
- **R10:** Methods are cost-gated, not rules
- **R11:** No historical-graphics emulation in LLM core
- **R12:** Tags are intent, not evidence — `V12.FUSE`, `V12.ELIDE`, etc. label intended method only

## File Creation Protocol

Before creating any `src/` file, check for the clean-room source register in the policy document. The clean-room note states: Cell mechanisms are described in academic papers (arXiv 0710.2442, arXiv 0910.2324, FSU TRs, UT-CS-07-597, Sequoia SC/PPoPS papers, UT Apollo paper, Intel Xe docs). IBM patents on Cell compilers/DMA ordering exist but are not needed for design.

If exposure to any Class C source occurs, record it immediately in the exposure ledger and move affected modules to `quarantine/`.