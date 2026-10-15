# Clean-Room Exposure Ledger

## Entry #001

- **Date:** 2026-10-15
- **Actor:** user (project owner)
- **Actor Type:** human
- **Item Viewed:** https://github.com/arun-prasath2005/gemma4-cpu-moe
- **Class:** Class D (Conditional - approved with restrictions)
- **Affected Modules:** MoE dispatch mechanism research, CPU memory management
- **Action:** Approved for mechanism study only - NOT for code copying
- **Reviewer:** Project owner (explicit verbal approval recorded)
- **Scope:** Understanding MoE CPU dispatch mechanism for developing similar approach in OpenVINO
- **Reasoning:** No academic papers available for CPU MoE optimization; need to understand mechanism to derive clean-room implementation
- **Restrictions:**
  1. No code copying or translation
  2. No import of code identifiers
  3. No reproduction of logic structure
  4. Only conceptual understanding to inform clean-room derivation
  5. All resulting code must cite Class A papers or derive independently from spec
- **Taint Status:** Modules using insights from this repo marked as "Class D informed" and require clean-room re-derivation from academic papers before production

## Entry #002

- **Date:** 2026-10-15
- **Actor:** agent:claude@opencodo-vpu-session-001
- **Actor Type:** AI agent
- **Item Viewed:** Local filesystem paths containing username "rina0423" (development environment)
- **Class:** N/A (Local environment access, not a source)
- **Affected Modules:** build.bat, CMakeLists.txt
- **Action:** Documented environment-specific paths in build artifacts
- **Reviewer:** N/A
- **Scope:** Local development environment paths only
- **Reasoning:** Build scripts reference local OpenVINO SDK path and VS BuildTools path
- **Restrictions:** None - these are non-sensitive filesystem paths
- **Exposure Risk:** LOW - paths are local dev environment, no secrets or credentials
- **Taint Status:** No taint - clean-room status of all src/ files unaffected