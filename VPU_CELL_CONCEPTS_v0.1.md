# VPU and Cell: Concepts Worth Borrowing, and Hacks Beyond Cell v0.1

**Companions:** `VPU_CONCEPT_v0.1.md` (methodology, P1–P14, H0–H7), `VPU_KERNEL_HARNESS_AND_TRACE_v0.1.md` (harness and trace), `VPU_CLEANROOM_POLICY_v0.1.md` (clean room)
**Status:** proposal. Nothing here enters the methodology until its hypotheses pass and the user approves it.
**Tags:** [V] verified in a source during this pass. [S] slides, lecture notes, or secondary analysis; a lead only. [M] from memory; verify before use. [PROPOSED] not yet adopted.

---

## 0. Bottom line

- Cell is the same design family as the VPU methodology: small explicit local memory, asynchronous DMA, many simple cores, and a bandwidth-first view. Much of it transfers directly.
- What hurt Cell was the programmer burden, not the core ideas. Hand-written DMA schedules, alignment rules, and local-store overflow were the cost. The hacks below move that burden into the kernel contract, a schedule generator, the harness, and the trace.
- **Drop:** SPE-to-SPE DMA as a general primitive, hand scheduling of the two issue pipelines, and sharing one local store between code and data.
- **Five new principle candidates** (P15–P19) and six hacks (§5). None adopted yet.
- **Main risk:** the signalling and placement hacks depend on memory ordering and device behaviour that has not been measured on the L14. Each one has a kill condition in §7.

---

## 1. Why Cell fits this methodology

Principle P3 (staged local store) came from the PS2 vector units. The Cell SPE is the same idea at larger scale: 256 KB of software-managed local store per SPE, DMA as the only way in and out, and a fixed grid of simple cores. The GPU analogue is the workgroup's shared local memory. On the L14 that is Intel's shared local memory (SLM), whose size must be queried, not assumed (§2, last row).

Cell is also a case study of what goes wrong with explicit management. Its programming burden is the reason the methodology needs a contract, a generator, and a harness.

---

## 2. Verified Cell facts used in this document

| Fact | Value | Source | Status |
|---|---|---|---|
| Cores | 1 PPE + 8 SPE | arXiv 0710.2442 | [V] |
| SPE local store | 256 KB, code and data share it | arXiv 0910.2324 | [V] |
| SPE registers | 128 registers × 128 bits | arXiv 0910.2324; INF5063 slides | [V] |
| Issue | dual-issue, in-order, two instructions per cycle | arXiv 0710.2442 | [V] |
| Peak single precision | 8 SPEs × 8 ops × 3.2 GHz = 204.8 GFLOP/s | arXiv 0710.2442 | [V] |
| EIB | 204.8 GB/s theoretical; about 197 GB/s achieved core-to-core (IBM figure) | arXiv 0910.2324; INF5063 slides | [V] / [S] |
| Per-SPE link | 25.6 GB/s in and out, theoretical | FSU TR-080906 | [V] |
| Main memory | XDR, 25.2–25.6 GB/s; sources differ | arXiv 0910.2324; INF5063 slides | [V] |
| Measured SPE↔memory | about 7 GB/s in one study's configuration | FSU TR-070724 | [V], condition-specific |
| Contention | with many SPEs transferring together, each sustains under 4 GB/s | FSU TR-080906 | [V] |
| Placement | thread-to-SPE assignment changes inter-SPE bandwidth | FSU TR-080906 | [V] |
| DMA queue | 16 outstanding requests per SPE; up to 128 in total on the EIB | FSU TR-080906; FSU TR-070724; UT-CS-07-597 | [V] |
| DMA size | up to 16 KB per request | FSU TR-080906; FSU TR-070724 | [V] |
| DMA alignment | multiples of 16 B; 128 B alignment is best | INF5063 slides; Cell best-practice slides | [S] |
| DMA ordering | tags; a fence orders DMAs that share a tag | FSU TR-070724 | [V] |
| Mailbox | inbound 4 × 32-bit entries; outbound depth 1 | arXiv 0910.2324 (inbound); outbound [M] | [V] / [M] |
| Branches | avoid on the SPE; static branch hints | INF5063 slides | [S] |
| Burden | "Cell's design is simple and causes software complexity" | UT Apollo paper | [V] |
| Missing feature | dynamic task-to-core mapping for load balancing was lacking in the Cell models compared | PPoPP 2009 | [V] |
| Working sets | Sequoia: tasks declare their data at each memory level, decomposed as a tree | SC 2006; PPoPP 2007 | [V] |
| SLM (Intel GPU) | per-Xe-core, program-managed, consistency is the program's job; size varies by generation | Intel Xe optimisation guide | [V] |

**Note on the SLM row.** Intel's guide lists per-Xe-core shared memory of 64 KB or 128 KB depending on the architecture, with a per-work-group maximum that can be lower. The L14's Core Ultra 5 135H uses Xe-LPG. Its value must be queried at runtime and written to `meta.json`. This document does not assume it.

---

## 3. Cell's failure modes (and what each one costs us)

1. **Hand-written DMA schedules.** Double buffering was rewritten per kernel. → Generate it from the contract (H-A).
2. **Alignment errors.** Misaligned DMA fails silently or corrupts data. → Reject misaligned descriptors in the manifest and the harness (H-G).
3. **Local-store overflow and code overlays.** Code and data share 256 KB. → Declared local and code budgets, with guard arenas (H-F, P19).
4. **Placement-sensitive bandwidth.** The same traffic runs at different speeds depending on which SPE does it. → Placement sweep (H-C, P16).
5. **No dynamic load balancing.** Work stays where it was first assigned. → Bounded claim queue (H-E, P18).
6. **Measured versus theoretical gaps.** SPE↔memory measured about 7 GB/s against 25.6 GB/s theoretical. → Same rule as H0: measure the achievable number before using it as a roofline.

---

## 4. Verdicts on Cell concepts

| Cell concept | Verdict | Reason and link |
|---|---|---|
| Software-managed local store (no cache) | **Adopt** | Already P3. Shared memory on the GPU is the analogue. |
| Asynchronous DMA with overlap (double or triple buffering) | **Adopt, generated** | H-A. The hand-written part is what fails. |
| DMA alignment rules | **Adopt as a harness check** | H-G. Cheap to check, and silent when violated. |
| Sixteen outstanding DMAs per SPE | **Adapt** | Becomes a queue-depth sweep parameter, measured per device. Not assumed. |
| DMA tags and fences | **Adapt** | H-B. Timeline values can play the same role as tags. |
| Mailboxes (small messages to and from the CPU) | **Adapt, carefully** | H-D. A progress word in host-visible memory, with memory-ordering tests. |
| SPE-to-SPE DMA (local store to local store) | **Drop as a primitive** | GPU workgroups cannot read each other's shared memory. Halo exchange through global memory is the fallback, only where a kernel needs it. |
| EIB topology and placement effects | **Adapt** | H-C. Test dispatch and tile order, not a model of the ring. |
| Dual-issue pipelines and hand scheduling | **Drop** | The GPU compiler schedules. Disassembly is for diagnosis, not for hand scheduling. |
| Branch avoidance and hints | **Already covered** | P8 (predication). |
| 128 × 128-bit register file | **Adapt as a budget** | Declared registers per invocation, under P19. |
| Code and data sharing local store; overlays | **Adapt as a code budget** | H-H, under P19. Shader variants are the code footprint. |
| Missing dynamic task mapping | **Adopt, bounded** | H-E. Persistent kernels stay deferred, as the triage says. |
| Programmer burden | **Lesson, not a feature** | Mitigated by the generator, the harness, and the trace. Programmers should not hand-write what the contract already states. |
| Sequoia-style tasks with declared working sets | **Adopt, already matched** | The kernel contract in methodology §1.3 is this idea. Sequoia's published work supports the approach. |
| PS3-style varying unit counts (yield) | **Adapt** | Query unit counts at runtime. Scheduling must tolerate missing or slower units. [M] for the PS3 detail. |
| Local store as the only memory | **Not applicable** | GPUs have global memory and caches. Cell's model is one special case. |

---

## 5. Hacks beyond Cell

Each hack has four parts: the Cell pain it targets, the mechanism, the test, and the kill condition. Vulkan and HIP mappings are [M] until checked against the spec for the target driver.

### H-A. Generated staging schedule from the contract

- **Pain:** hand-written double and triple buffering, rewritten per kernel (failure mode 1).
- **Mechanism:** the kernel contract (methodology §1.3) already declares tile footprints. A generator emits the staging loop with a ring of slots, depth 1, 2, or 3 from the manifest's sweep. Kernel authors write the compute body only.
- **Vulkan mapping:** shared memory slots, with loads issued one slot ahead. Vulkan has no direct equivalent of CUDA's `cp.async` in core compute [M], so overlap comes from software pipelining. A transfer queue may provide the overlap where the device exposes one [M].
- **Test:** generated versus hand-tuned schedule on the large F32 matvec. Also count edge-suite failures in each.
- **Kill:** generated schedule more than 5% slower than the tuned hand version at large shapes, after the depth sweep.
- **Principle:** P15.

### H-B. Timeline values as DMA tags

- **Pain:** Cell's tag groups let a kernel wait for a subset of transfers. Vulkan fence pools are costly when there are many small transfers.
- **Mechanism:** one timeline semaphore per transfer stream. Each slot waits for a value; the value encodes slot and generation.
- **Test:** host time per transfer, fence pool versus timeline, at 1,000 or more transfers per second.
- **Kill:** no reduction in host time at that rate.
- **Principle:** P15 (sub-point).

### H-C. Placement-aware dispatch order

- **Pain:** placement changes bandwidth (failure mode 4).
- **Mechanism:** a sweep over workgroup dispatch order and tile-to-address mapping (linear, swizzled, and blocked). Each order is a variant in the manifest.
- **Test:** achieved GB/s across orders on the large F32 matvec. Report the spread.
- **Kill:** spread under 5% across all orders. Then drop P16 for this device.
- **Principle:** P16.

### H-D. Progress word instead of mailbox

- **Pain:** per-token fences on the CPU add a wait each token. Cell used small mailbox messages for this.
- **Mechanism:** the GPU writes a progress counter, or the sampled token ID, into host-visible memory. The CPU polls with backoff. The data behind a progress word must become visible before the word is read. That needs a release and acquire pair, ideally a semaphore or fence, not the word alone [M].
- **Test:** time to first visible token and inter-token latency, against per-token fences. Also a litmus test (§8.2) that checks for stale reads across at least 10⁶ iterations.
- **Kill:** any stale read in the litmus test, or polling cost above 5% of one CPU core while waiting, or no latency gain.
- **Principle:** P17. Gated on the litmus test passing.

### H-E. Bounded dynamic claim queue

- **Pain:** no dynamic load balancing (failure mode 5).
- **Mechanism:** the host fills a bin table for the dispatch. Workgroups claim bins with an atomic counter in global memory. This is per dispatch, not a persistent kernel, so it stays inside the triage's deferral. Persistent queues remain deferred.
- **Test:** skewed MoE trace (H7 in the concept doc): static assignment versus claim. Compare the tail of the batch time.
- **Kill:** under 20% tail improvement at skew of 2× or more, or any loss on uniform work above 3%.
- **Principle:** P18.

### H-F. Declared local budgets with guard arenas

- **Pain:** local-store overflow, which Cell reported only as corruption (failure mode 3).
- **Mechanism:** each workgroup gets an arena with a declared budget in the manifest. The harness rejects a kernel whose declared usage exceeds the queried limit. Debug builds place a guard region after each arena and check it at batch end. This extends the `canary` debug check in the harness document.
- **Test:** inject an overflow in a test kernel. The harness must reject it before the run, and the debug build must catch it at runtime.
- **Kill:** none. This is a correctness check. Its cost is measured in the overhead test.
- **Principle:** P19.

### H-G. Alignment-typed descriptors (harness check)

- **Pain:** misaligned DMA, from failure mode 2.
- **Mechanism:** manifest validation rejects tile sizes and base offsets that break the target's required alignment. The edge suite adds deliberately misaligned base offsets.
- **Test:** count silent misalignment faults before and after the check.
- **Kill:** none. Cheap and always on.
- **Principle:** none. This is a harness rule.

### H-H. Code-footprint budget (overlay lesson)

- **Pain:** Cell code overlays, where code and data compete for one small store (failure mode 3).
- **Mechanism:** count compiled pipeline and shader variants per kernel. Cap them in the manifest. The trace already logs `compile.pipeline` events, so first-token latency can be checked against variant count.
- **Test:** cold-start time against variant count across the sweep grid.
- **Kill:** no measurable cold-start effect. Then keep the cap for maintainability only.
- **Principle:** P19 (shared with H-F).

---

## 6. Principle candidates

| ID | Name | From | Status |
|---|---|---|---|
| P15 | Generated staging schedules, with timeline-value slot waits | H-A, H-B | [PROPOSED] |
| P16 | Placement-aware dispatch order | H-C | [PROPOSED] |
| P17 | Progress-word signalling, gated on a litmus test | H-D | [PROPOSED] |
| P18 | Bounded dynamic claim per dispatch | H-E | [PROPOSED] |
| P19 | Declared local and code budgets, with guard arenas | H-F, H-H | [PROPOSED] |

Existing principles touched: **P3** gains the generator (P15). **P10** gains the claim mechanism (P18). **P12** gains the progress word as a control path (P17). None of these changes take effect until approved.

---

## 7. Hypotheses and kill conditions

Thresholds are proposals. Fix them before the first run. The numbering continues from H0–H7 in the concept document.

| ID | Hypothesis | Experiment | Pass (proposed) | Kill | Principle |
|---|---|---|---|---|---|
| H8 | Generated staging matches hand-tuned | H-A on F32 matvec, depth sweep, plus edge suite | within 5% at large shapes; fewer edge failures | more than 5% slower after the sweep | P15 |
| H9 | Timeline slot waits cut host cost | H-B, many small transfers | at least 20% less host time per transfer | no reduction at 1,000 per second or more | P15 |
| H10 | Dispatch order changes bandwidth | H-C, orders × large F32 matvec | spread of at least 5% across orders, with a clear best order | spread under 5% | P16 |
| H11 | Progress word reduces per-token wait | H-D, against per-token fences | at least 30% less wait per token; litmus test passes | any stale read; polling above 5% of a core; no gain | P17 |
| H12 | Bounded claim helps skewed work | H-E, skewed MoE trace | at least 20% better tail; no more than 3% loss on uniform work | under 20% tail gain at 2× skew | P18 |
| H13 | Code footprint affects cold start | H-H, variant count against cold-start time | a measurable slope | no slope | P19 |

**Order.** H8 first, because it tests whether the contract can generate schedules at all. H10 next, because it is cheap. H11 and H12 come only after the litmus test is written and passing.

---

## 8. Harness and trace changes

### 8.1 Trace schema v2 (proposed)

Adds columns to `spans.csv`, and bumps `schema_version` to 2. Old readers must refuse v2 files, as the harness document requires.

- `queue_kind`: `compute` or `transfer`
- `slot_id`, `slot_generation`: for staged transfers (H-A, H-B)
- `dispatch_order_id`: the variant's order (H-C)
- `claim_attempts`, `claim_contention`: for claim queues (H-E)
- `progress_polls`, `progress_stale`: for progress words (H-D). `progress_stale` must be zero.
- `slm_bytes_declared`, `slm_bytes_limit`: the declared budget and the queried limit (H-F)
- `variant_count_loaded`: for code footprint (H-H)

New span names: `sig.progress` (H-D), `sched.claim` (H-E), `place.order` (H-C, zero-duration event at dispatch).

### 8.2 New self-tests

- **Litmus test for progress words (H-D).** A writer stores data, then a release-ordered progress word. A reader spins on the word with acquire ordering, then checks the data. Run at least 10⁶ iterations, on the target device and driver. Any stale read fails the test, and H-D stays blocked until it passes.
- **Placement sweep (H-C).** Runs as a self-test per device, not as a one-off experiment. Results go to `meta.json`.
- **Limit query.** Records `maxComputeSharedMemorySize` [M] and the queue family list, so every SLM and transfer claim is checked against recorded values.

---

## 9. Backend mapping notes

- **Vulkan.** Shared memory slots (P15). Timeline semaphores (H-B) [M]. Transfer queue families (P14 and H-A) [M], probed at runtime. Host-coherent memory for progress words (H-D) [M]. Shared-memory limit query (H-F) [M]. Atomic counters for claims (H-E) [M].
- **HIP.** Local data share as the slot memory (P15). Streams for the transfer pattern. Atomics for claims. Not on the L14's target path, so this stays a mapping note.
- **Intel SLM.** Program-managed, with consistency as the program's job. Size per Xe-core must be queried (§2).

---

## 10. Clean-room source register for this document

**Class A (allowed, cite by section):**
- arXiv 0710.2442 (QCD on the Cell BE): architecture figures
- arXiv 0910.2324 (matrix languages on the Cell BE architecture): EIB, mailbox, local store
- FSU technical reports TR-070724, TR-080906, TR-061215: DMA, EIB, placement
- UT-CS-07-597: DMA and local-store rules
- Sequoia (SC 2006; PPoPP 2007, 2008) and PPoPP 2009 (comparing programming models for multiprocessors)
- UT Apollo paper: programmability burden
- Intel Xe architecture and SLM optimisation guides (vendor documentation)

**Class D (lead only; verify against a Class A source before use):**
- INF5063 Cell slides; the Cell best-practice slides; the MIT Lincoln Lab HPEC 2006 slides
- Chips and Cheese Xe and Battlemage analysis articles
- Gaps noted in §2 (outbound mailbox depth, PS3 unit count)

**Not used:**
- Cell SDK sample code and tooling
- IBM patents on Cell compilers and DMA ordering (found in search; they describe compiler internals and are not needed for design)

**Clean-room note.** Cell mechanisms are described in academic papers. Patents on Cell mechanisms may exist, so the commercial-release check in the policy (§8) applies.

---

## 11. Open decisions

1. **Approve the candidates?** P15–P19 stay proposed until H8–H13 run. Confirm the order in §7.
2. **Progress words (H-D).** Keep them, given the memory-ordering risk? Or drop them until the litmus test is designed?
3. **Persistent kernels.** H-E stays bounded (per dispatch). Confirm that this is within the triage's deferral.
4. **Intake answers still pending.** Language and OS affect H-A (generator target), H-B (timeline semantics per API), and H-F (limit queries). These items wait on those answers.
