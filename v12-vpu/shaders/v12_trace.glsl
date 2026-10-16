// V12 reusable in-kernel perf tracing (gated by -DV12_TRACE).
//
// Purpose: show which REGION of a kernel consumes the hardware time, not just
// the whole-dispatch time. Four regions are recorded per invocation:
//   0 = decode   (unpack raw weight block bytes -> integer code + scale/min)
//   1 = mac      (integer multiply-accumulate + activation byte loads)
//   2 = scale    (per-group scale/min application / bookkeeping)
//   3 = reduce   (subgroup/workgroup reduction + store)
//
// Mechanism: GL_ARB_shader_clock realtime counter read at region boundaries; the
// per-invocation tick deltas are summed and written to a trace buffer. The host
// converts ticks -> us with device timestampPeriod and reports the share per
// region. Timing is per representative invocation (wall clock), so region shares
// are what matter; absolute sums are SIMD-inflated.
//
// When V12_TRACE is undefined the helpers collapse to constants, so the compiler
// dead-code-eliminates all tracing and the shipped kernel is unchanged.
//
// Provenance (AGENTS.md): clean-room. clockRealtime2x32EXT is the standard GLSL
// (GL_ARB_shader_clock) device timer primitive, used here as a measurement tool.
#ifndef V12_TRACE_GLSL
#define V12_TRACE_GLSL

#ifdef V12_TRACE
layout(set=0, binding=4, std430) buffer V12TraceBuf { uint v12tr[]; };
uvec2 v12_now() { return clockRealtime2x32EXT(); }
uint  v12_dt(uvec2 a, uvec2 b) { return b.x - a.x; }
#define V12_TRACE_STORE(idx, dec, mac, scl, red) do { \
    uint _b = (idx) * 4u; \
    v12tr[_b + 0u] = (dec); v12tr[_b + 1u] = (mac); \
    v12tr[_b + 2u] = (scl); v12tr[_b + 3u] = (red); \
} while (false)
#else
uvec2 v12_now() { return uvec2(0u); }
uint  v12_dt(uvec2 a, uvec2 b) { return 0u; }
#define V12_TRACE_STORE(idx, dec, mac, scl, red) do {} while (false)
#endif

#endif
