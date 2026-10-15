# VPU Clean-Room Policy and Enforcement v0.1
**Scope:** every module, kernel, spec entry, test, and document written under the VPU methodology
**Rule in one line:** we build from papers. We may name a mechanism that a paper describes. We do not copy, translate, paraphrase-by-porting, or regenerate code from any LLM system.
**Companion docs:** `VPU_CONCEPT_v0.1.md` (methodology), `VPU_V4_IMPLEMENTATION_TRIAGE.md` (triage)

---

## 0. Intake: ask before any work

**Do not create any file under `src/` until the four core answers below are recorded in §0.2 with status Confirmed.** Ask these in order. If the conversation is interactive, ask one at a time. Do not fill any answer from defaults or from examples in other project documents.

**Ask user for:**
1. **Hardware target.** CPU, GPU, memory type and size, and which machine runs the benchmarks.
2. **OS target.** Windows, Linux, or both, with versions.
3. **Programming language target.** Host language and kernel language, for example C++ host with GLSL or SPIR-V kernels, or HIP C++, or OpenCL C. Do not assume one.
4. **GPU API targets.** Multi-select, at least one required. Options: Vulkan, OpenVINO, ROCm / HIP, CUDA, other (name it). No option is selected by default. For each selected API, also ask: the minimum version or level, the device or device class it must run on, and its priority order (§0.3).

**Also ask (supporting):**
5. Who writes the spec and who implements: humans, AI coding agents, or both.
6. May any forbidden system be run as a black-box output oracle? Default is **no** (see §9).
7. Which license should the output carry?

### 0.1 Question wording (exact)

> Before I start, I need four things. What is the hardware target (CPU, GPU, RAM, and the machine used for benchmarks)? What is the OS target? What is the programming language target for host code and for kernel code? Which GPU APIs should I build for? Choose any of Vulkan, OpenVINO, ROCm / HIP, CUDA, or others, and give a priority order.

### 0.2 Target record

| Field | Value | Source | Status |
|---|---|---|---|
| Hardware target | Not yet confirmed. Project records name an Intel Core Ultra 5 135H with Arc iGPU (128 EU) and 16 GB RAM, on a ThinkPad L14 | project records | **Confirm** |
| OS target | Not yet confirmed. Project benchmarks run on Windows | project records | **Confirm** |
| Language target | Not recorded | none | **Ask** |
| GPU API targets | Not selected. Multi-select: Vulkan, OpenVINO, ROCm / HIP, CUDA, other. No default. | none | **Ask** |
| API priority order | Not recorded | none | **Ask** |
| API ↔ hardware pairing | Not checked. Depends on the hardware target and the API selection (§0.3) | this doc | **Pending** |
| Implementers | Not recorded | none | **Ask** |
| Oracle policy | Default: no forbidden black-box oracles | this doc §9 | **Ask** |
| Output license | Not recorded | none | **Ask** |

**Gate rule:** a field can move to Confirmed only when the user has answered. A value from project records stays at Confirm until the user says yes.

**API gate rule:** at least one GPU API must be Confirmed. Each API starts work only after it is Confirmed. Its vendor docs and runtimes are enabled only then (§2.1, §2.2, §10.2).

### 0.3 GPU API selection and pairing

Choose APIs from the menu in item 4. Do not infer them from the examples in other project documents. The device scope below is the usual one for each API. Confirm each entry against the vendor documentation before relying on it.

| API | Device scope [M] | Needed from the user |
|---|---|---|
| Vulkan | any GPU with a Vulkan driver | minimum Vulkan version; required extensions |
| OpenVINO | Intel devices: CPU, GPU, NPU | which device(s) it targets; model and precision needs |
| ROCm / HIP | AMD GPUs | GPU architectures to target |
| CUDA | NVIDIA GPUs | GPU architectures (compute capability) to target |
| Other | — | name it, and state its device scope |

**Pairing rule.** Record every selected API against the hardware target. A mismatch is an open item. For example, CUDA selected while the hardware target has no NVIDIA GPU. Never substitute another API silently. On the recorded hardware, the CPU, the integrated GPU, and the NPU are separate devices [S], and OpenVINO's device choice among them is its own intake item.

---

## 1. Definitions

- **Clean room.** A development process in which every design decision and every line of code is derived from allowed sources, with records that show it.
- **Forbidden source (Class C).** Source code of an LLM system or its companion repos, listed in the denylist (§2, §10.1). No one reads it, copies it, translates it, or regenerates it.
- **Allowed source (Class A).** Papers in the paper map, vendor API and specification documents, and standards. Used with citations.
- **Consume-only (Class B).** Vendor runtimes and SDKs. We call their public APIs and link their binaries. We do not read or copy their implementation source unless their license and this policy both allow it.
- **Conditional (Class D).** Items that need a named, logged approval before use (§2.4).
- **Mechanism reference.** Naming a mechanism that a Class A paper describes, and citing that paper. Allowed.
- **Copy.** Any of the following, whatever the target language:
  - **C1 Verbatim.** A contiguous run of normalised tokens that matches a Class C source.
  - **C2 Structural.** Equal function structure after identifier and literal normalisation.
  - **C3 Translation.** A line-by-line or block-by-block port to another language. Changing the language does not make it a new work.
  - **C4 Constants and tables.** Magic numbers, codebooks, lookup tables, tuning tables, or thresholds that do not appear in a Class A source.
  - **C5 Names and text.** Identifiers, file names, comments, docstrings, or doc text taken from Class C.
  - **C6 Regeneration.** Producing Class C code from memory, by a person or an AI agent, whether or not the output looks identical.
- **Exposure.** Any event in which a person or agent opened, searched, quoted, or summarised Class C content. Exposure is recorded in the ledger (E6).
- **Taint.** A module is tainted when an exposure touches it or its spec. Tainted modules are quarantined (§7).

---

## 2. Source classes

### 2.1 Class A: allowed (cite by section)
- Papers in the paper map, cited by arXiv or DOI ID and section, equation, figure, or table.
- Vendor specifications for each API selected at intake (§0.2): Vulkan specification and guides, SPIR-V and GLSL specifications (Vulkan); HIP and ROCm documentation; OpenVINO documentation; CUDA documentation. An API that is not selected has no entries.
- Standards such as IEEE 754.
- Textbooks, cited.

### 2.2 Class B: consume-only
- For each selected API only: Vulkan loader, drivers, and validation layers; ROCm / HIP runtime; OpenVINO runtime; CUDA runtime; and the matching device drivers. Call their public APIs only.
- Consumption means linking or calling the public API. It does not mean reading or copying the implementation source. Headers shipped in an SDK may be included as the SDK's API surface, and only that.

### 2.3 Class C: forbidden (denylist, §10.1)
- Repos and packages of LLM inference and training systems. Initial seeds are listed in §10.1 and must be verified before enforcement.
- Their documentation pages, issue threads, pull requests, commit diffs, and code search results.
- Blog posts and forum answers that quote their code.
- Any fork, mirror, vendored copy, or snippet of the above.
- If the user asks for a specific git to be read and researched for ideas, it is allowed

**Mechanisms seen only in Class C are not allowed as a basis for design.** If a Class C system uses a technique, the technique may be used only if a Class A paper describes it. Cite the paper.

### 2.4 Class D: conditional (approval required)
- Companion code from a paper's authors. Default: forbidden. Approval requires a license check, a named repo, and a ledger entry.
- Public format specifications (for example, file format documents published by a model-format maintainer). Default: allowed only as a written spec copied into `spec/formats/` with its source URL and date, and only after approval.
- Any web page not on the Class A domain list.

Approval is logged in `CLEANROOM_LEDGER.csv` with: date, item, approver, scope (which section), and the reason it is needed.

---

## 3. Roles and separation

Strict separation is the default. A single person may play more than one role only if their exposure record allows it.

| Role | May read | May not read | Output |
|---|---|---|---|
| Spec writer | Class A, Class B docs | Class C, Class C-derived notes | spec entries with paper citations |
| Implementer | spec, Class A, Class B docs | Class C, any web page not on the allowlist | code with provenance headers |
| Auditor | Class C corpus snapshot, via scanners only | nothing forbidden to them, but they should not forward excerpts | scan reports listing locations and hashes, never excerpts |
| Reviewer | provenance headers, scan reports, spec | Class C | review record |

**Auditor rule.** The auditor holds the Class C snapshot on a machine the implementers cannot access. Reports contain file path, line range, match length, and the snapshot hash. They do not contain forbidden text. If a human must see forbidden text to resolve a hit, that human becomes exposed and their modules are tainted.

**Exposure reset.** A person who has been exposed may still act as spec writer for mechanisms that are not in the exposed area. They may not implement the affected mechanism, and they may not review the affected module.

---

## 4. Provenance

### 4.1 Header block (required in every source file under `src/`)

```text
// VPU-CR-PROVENANCE
// Mechanism:  <principle id, e.g. P3 staged local store>
// Derived-from: <arXiv or DOI id>, <section / equation / figure>
// Spec entry: spec/<file>#<anchor>
// Author:     human:<initials> | agent:<model id>@<session id>
// Exposure:   none
```

### 4.2 Commit trailers (required for any commit touching `src/`)

```text
Clean-Room-Source: <arXiv or DOI id>, <section>
Clean-Room-Exposure: none
```

A commit with any exposure must not touch `src/`. It goes to `quarantine/` (§7).

### 4.3 Spec entries

Each spec entry states:
- the mechanism, with its Class A citation,
- the algorithm, written from the paper's own notation,
- every parameter, with the paper line it came from,
- an **UNSPECIFIED-BY-PAPER** list for anything the paper does not settle (§6).

---

## 5. Enforcement controls

Each control lists what it checks, when it runs, and what it cannot catch.

### E1. Denylist scan (pre-commit and CI)
- **Checks:** forbidden repo names, package names, identifier prefixes, and path names in code, comments, file names, and docs in the repo.
- **Runs:** pre-commit on staged files; CI on the full tree.
- **Cannot catch:** copies that use none of the listed strings.
- **Note:** this policy names only public API prefixes, not internal identifiers of Class C systems. Adding internal identifiers would put them into the repo.

### E2. Fingerprint overlap (CI, auditor corpus)
- **Checks:** normalised token k-grams with winnowing fingerprints (Schleimer et al., 2003 [M]) against the Class C snapshot.
- **Parameters (initial, tune on a known-copy test set):** normalise identifiers to placeholders, drop comments and whitespace, k = 10 tokens, winnow window = 4.
- **Fail rule:** any shared fingerprint covering 12 or more normalised tokens fails CI.
- **Review rule:** shorter shared runs go to a human reviewer, with an allowlist for generic idioms (loop headers, bounds checks).
- **Cannot catch:** heavy reordering or re-expression.

### E3. Structural hash (CI, auditor corpus)
- **Checks:** normalised function-level syntax hashes, with identifiers and literals replaced.
- **Fail rule:** a matching hash on a function of 8 or more statements fails CI.
- **Cannot catch:** structures re-written by hand with different control flow.

### E4. Constants and tables scan (CI, auditor corpus)
- **Checks:** numeric arrays of 16 or more entries, codebooks, lookup tables, and tuning constants that match Class C content.
- **Fail rule:** any match fails CI. Constants that also appear in a Class A paper are exempt, and the exemption must cite that paper.

### E5. Provenance gate (commit hook)
- **Checks:** commits that touch `src/` contain the two trailers in §4.2 with `Clean-Room-Exposure: none`.
- **Sketch:** see §10.3.
- **Cannot catch:** a false claim of independence. That is handled by E10.

### E6. Exposure ledger
- **File:** `CLEANROOM_LEDGER.csv`. Append-only. Changes are reviewed.
- **Fields:** `date, actor, actor_type (human|agent), item_viewed, class, affected_modules, action, reviewer`.
- **Rule:** any exposure marks the affected modules tainted until they are rewritten from spec (§7).

### E7. Dependency and vendoring gate
- **Checks:** no vendored source code, and no submodules, except for SDKs listed in §2.2.
- **Rule:** a new dependency requires a Class A or Class B entry and a ledger line.
- **Output:** a software bill of materials, regenerated on each release.

### E8. AI coding agent controls
AI coding agents are a clean-room risk. They may have seen Class C code during training, and they may reproduce it from memory. These controls apply to every agent session.
- **a. Instructions.** The agent's standing instructions prohibit reproducing or recalling Class C code, and require a Class A citation for every function it writes.
- **b. Workspace.** The agent's workspace contains no checkout of any Class C repo, and no Class C files.
- **c. Network.** Agent web access uses the allowlist in §10.2 only.
- **d. Output gate.** Every agent-written file passes E1 to E4 before it is committed.
- **e. Records.** The agent model ID and session ID are written into the provenance header (§4.1).
- **f. Limit.** The agent's memory cannot be inspected. E2 to E4 are the backstop. A hit from them is not fixed by editing. It is fixed by re-deriving the code from spec in a new session that has not seen the flagged output (§7).

### E9. Test oracle policy
- **Default:** tests compare against independent reference implementations written from Class A papers (for example, a naive matrix-vector reference). The triage's CPU reference follows this rule.
- **Black-box forbidden binaries:** not allowed by default. If approved (§9), the binary is run only to produce outputs. Its source, docs, and logs are never opened. Each use is logged.

### E10. Audits
- **Monthly:** full scan with E1 to E4 against the current tree.
- **Sampling:** pick three modules at random. The auditor checks each header (§4.1), and the spec entry it cites. Then a second implementer re-derives one function from its spec, in a new session, and the result is compared with the original.
- **Failure:** a sample that cannot be re-derived from its spec means the spec is incomplete. Fix the spec. Do not patch the code to match a guess.

---

## 6. Gaps: when the paper does not say enough

This is the point where people and agents are most tempted to look at a Class C implementation. The procedure below replaces that.

1. Write the gap as an **UNSPECIFIED-BY-PAPER** entry in the spec. State what is missing and which parameter is affected.
2. Resolve the gap by **experiment**: a sweep, a sensitivity test, or a comparison of two candidates against the Class A reference. Record the result as a new spec entry with its evidence.
3. If experiments cannot resolve it, request **Class D approval** for a named source and a named section, and log it (§2.4).
4. Never "check how it is done" in a Class C repo. That is an exposure (§1), whatever the reason.

---

## 7. Violation response

| Level | Trigger | Action |
|---|---|---|
| L1 Review | E2 or E3 hit below the fail threshold; E4 near match | Human reviewer decides. Record the decision and the reasoning in the ledger. |
| L2 Quarantine | E1 hit, or E2 or E3 at or above threshold, or any E4 match | Move the module to `quarantine/`. Taint it. Re-derive it from spec in a new session by an implementer who has not seen the flagged output. Re-scan. |
| L3 Confirmed | Verbatim or identifier leak; confirmed exposure; a false provenance claim | Quarantine. Re-derive. Notify the user. Decide the history rewrite policy for any already-pushed commit (a user decision). |

**Renaming is not a fix.** Changing identifiers, reformatting, or reordering a copied function does not make it clean. Re-derive from spec.

---

## 8. What this policy does and does not guarantee

- **Scans** detect verbatim, near-verbatim, and some structural copies. They do not prove independent derivation.
- **Paper-only** is a process rule. It is enforced through provenance headers, commit trailers, the exposure ledger, and audits. It is not something code analysis can prove.
- **Memorisation.** AI models may have memorised Class C code. The scans are a backstop and do not remove the risk.
- **Copyright and patents.** Copyright generally protects expression, not methods. Some methods in papers may be patented, and a clean room does not address patents. Check each paper's license and patent position before any commercial release. This policy is an engineering control and is not legal advice. Get counsel before release.

---

## 9. Open decisions (for the user)

1. **Oracle policy.** Can any forbidden system be run as a black-box oracle? Default is no.
2. **Class D items.** Are public format specifications allowed as written specs? Are paper companion repos ever allowed?
3. **Auditor.** Who holds the Class C snapshot, and on what machine?
4. **Output license.** Which license?
5. **History policy.** If an exposure reaches a pushed commit, should history be rewritten, or should the module be replaced forward?

---

## 10. Appendix: sample configuration

### 10.1 `cleanroom/denylist.yaml` (Class C, read-prohibited)

```yaml
# Seeds only. Verify each slug before enforcement. Owner maintains this list.
repos:
  - ggml-org/llama.cpp
  - ggml-org/ggml
  - vllm-project/vllm
  - sgl-project/sglang
  - NVIDIA/TensorRT-LLM
  - mlc-ai/mlc-llm
  - turboderp/exllamav2
  - ollama/ollama
  - huggingface/text-generation-inference
  - ml-explore/mlx-lm
  - huggingface/candle
identifier_prefixes:        # public API prefixes only
  - ggml_
  - GGML_
  - llama_
  - LLAMA_
path_patterns:
  - "**/ggml*"
  - "**/llama.cpp*"
```

### 10.2 `cleanroom/allowlist.yaml` (Class A and B)

```yaml
papers:                     # cite by section in every use
  - id: "<arXiv id from the paper map>"
    sections: ["<section list>"]
gpu_apis:                   # enabled only when the API is Confirmed at intake (§0.2)
  vulkan:   {selected: pending, docs: [https://docs.vulkan.org/, https://registry.khronos.org/SPIR-V/]}
  openvino: {selected: pending, docs: ["<OpenVINO documentation URL: confirm before enabling>"]}
  rocm_hip: {selected: pending, docs: ["<ROCm / HIP documentation URL: confirm before enabling>"]}
  cuda:     {selected: pending, docs: [https://docs.nvidia.com/cuda/]}
  other:    {selected: pending, docs: []}
web_domains:                # agent and dev web access: arxiv.org always; other domains only when their API is selected
  - arxiv.org
  # add docs.vulkan.org, registry.khronos.org, and the other API domains only when that API is selected
```

### 10.3 `.git/hooks/commit-msg` (sketch)

```bash
#!/usr/bin/env bash
# Reject commits that touch src/ without clean-room trailers.
msg=$(cat "$1")
if git diff --cached --name-only | grep -q '^src/'; then
  echo "$msg" | grep -q '^Clean-Room-Source: ' \
    || { echo "missing Clean-Room-Source trailer"; exit 1; }
  echo "$msg" | grep -q '^Clean-Room-Exposure: none$' \
    || { echo "src/ commits require Clean-Room-Exposure: none; exposed work goes to quarantine/"; exit 1; }
fi
```
