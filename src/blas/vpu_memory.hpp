// VPU-CR-PROVENANCE: author=deploy3-shardC file=vpu_memory.hpp
// Clean-room origin: this project (V12-VPU clean-room).
// Memory layout operations: scale+copy, transpose copies, and
// alpha*A + beta*B epilogue (geadd).
//
// R9: every method is an independent ablation switch.
// References: Lawson et al. (1979) §6; Golub & Van Loan (2013) §2.5.

#pragma once
#include "vpu_blas.hpp"

namespace Vpu {

// ── somatcopy: y = alpha * x  (out-of-place, same layout) ─────────────
// V12.TILE + V12.FUSE: SLM tile; fuse scale into the copy kernel.
VpuResult somatcopy(int n, float alpha,
                    const float* x, int incx,
                    float* y, int incy,
                    const MemoryConfig& cfg, DispatchContext* ctx);

VpuResult domatcopy(int m, int n, float alpha,
                    const float* a, int lda,
                    float* b, int ldb, char trans,
                    const MemoryConfig& cfg, DispatchContext* ctx);

// ── simatcopy: in-place y = alpha * x ──────────────────────────────────
// V12.TILE: in-place via SLM buffer, 32×32 tile.
VpuResult simatcopy(int m, int n, float alpha,
                    float* a, int lda, int ldc, char trans,
                    const MemoryConfig& cfg, DispatchContext* ctx);

// ── dimatcopy: in-place transpose + scale ──────────────────────────────
// V12.TILE: in-place via SLM buffer, 32×32 tile.
VpuResult dimatcopy(int m, int n, float alpha,
                    float* a, int lda, int ldc, char trans,
                    const MemoryConfig& cfg, DispatchContext* ctx);

// ── geadd: C = alpha*A + beta*B  (fused epilogue) ──────────────────────
// V12.FUSE + V12.TILE: alpha*A + beta*B in one kernel.
// Hypothesis H-GEADD-FUSE-1: win when 2× reads < 1 fused kernel reads.
VpuResult geadd(int m, int n, float alpha, const float* a, int lda,
                float beta, const float* b, int ldb,
                float* c, int ldc,
                const MemoryConfig& cfg, DispatchContext* ctx);

// ── CPU references (R4) ────────────────────────────────────────────────
void cpu_somatcopy_ref(int n, float alpha,
                       const float* x, int incx, float* y, int incy);

void cpu_domatcopy_ref(int m, int n, float alpha,
                       const float* a, int lda, float* b, int ldb, char trans);

void cpu_simatcopy_ref(int m, int n, float alpha,
                       float* a, int lda, int ldc, char trans);

void cpu_dimatcopy_ref(int m, int n, float alpha,
                       float* a, int lda, int ldc, char trans);

void cpu_geadd_ref(int m, int n, float alpha, const float* a, int lda,
                   float beta, const float* b, int ldb,
                   float* c, int ldc);

// ── Contract builder (R3) ──────────────────────────────────────────────
WorkloadContract make_memory_contract(const char* op, int m, int n,
                                      DType dt, Layout layout,
                                      int lda, int ldb, int ldc,
                                      int incx, int incy,
                                      MethodSet methods, Backend backend,
                                      const MemoryConfig& cfg);

} // namespace Vpu
