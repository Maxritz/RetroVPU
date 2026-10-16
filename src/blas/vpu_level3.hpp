// VPU-CR-PROVENANCE: author=deploy3-shardC file=vpu_level3.hpp
// Clean-room origin: this project (V12-VPU clean-room).
// Level 3 BLAS declarations — implements the subset declared in vpu_blas.hpp.
//
// R9: every method is an independent ablation switch. A `combined` variant
//     is a candidate, never a default (AGENTS.md R9).
//
// References: Lawson et al. (1979) §6; Golub & Van Loan (2013) §15;
//   Demmel (1990) blocked algorithm.

#pragma once
#include "vpu_blas.hpp"

namespace Vpu {

// ── Tile autotuning grid for SLM 128×128×K ──────────────────────────
// Generated from the workload contract (R3) before timing.
struct TileConfig {
    int32_t M, N, K;           // current tile shape
    int32_t group_m, group_n;  // workgroups per tile
    int32_t best_M, best_N, best_K;  // autotuned selection
};

// Autotune over a modest tile grid; returns the best K for given M×N.
// Hypothesis H-GEMM-TILE-1: 128×128×64 tile; depth-2 staging matches throughput.
TileConfig tile_autotune_gemm(int32_t M, int32_t N, int32_t K,
                              const ComputeCell& cell);

// ── H-GEMM-FUSE-1: batch > 4 identical shapes → fuse beats separate ──
// Returns true if the shapes in the batch group are identical and fusion
// is cost-effective at the current batch size.
bool gemm_batch_should_fuse(int32_t batch_count, int32_t M, int32_t N, int32_t K);

// ── H-TRSM-ELIDE-1: CPU panel-64 vs GPU panel-64; threshold at 512 ───
// H-C dispatch-order sweep result.
struct TrsmRoute {
    Backend   backend;
    int32_t   panel_size;   // 64 for small, variable for GPU
    DispatchOrder order;
};
TrsmRoute trsm_route(int32_t M, int32_t N, int32_t K,
                     const Level3Config& cfg, const DispatchContext* ctx);

// ── H-SYRK-ELIDE-1: triangle compute saves 50%; GPU wins at N=2048 ───
bool ssyrk_route_gpu(int32_t N, int32_t K, const Level3Config& cfg);

// ── H-GEADD-FUSE-1: fused alpha*A+beta*B vs separate ─────────────────
// Win when 2× reads < 1 fused kernel's reads.
bool geadd_fuse_is_beneficial(bool tile, int32_t M, int32_t N);

// ──────────────────────────────────────────────────────────────────────
// Level 3 BLAS function declarations
// All return VpuResult with timing + correctness (R7, R4, R8).
// ──────────────────────────────────────────────────────────────────────

// SGEMM / DGEMM — V12.TILE + V12.FUSE + V12.PRECISION
// OpenVINO MatMul (Class B) path; SLM 128×128×K; INT8/BF16 via V12.PRECISION.
VpuResult sgemm(char transa, char transb, int m, int n, int k,
                float alpha, const float* a, int lda,
                const float* b, int ldb, float beta,
                float* c, int ldc,
                const Level3Config& cfg, DispatchContext* ctx);

VpuResult dgemm(char transa, char transb, int m, int n, int k,
                double alpha, const double* a, int lda,
                const double* b, int ldb, double beta,
                double* c, int ldc,
                const Level3Config& cfg, DispatchContext* ctx);

// SBGEMM — V12.PRECISION: BF16 compute, FP32 accumulate
VpuResult sbgemm(char transa, char transb, int m, int n, int k,
                 float alpha, const uint16_t* a, int lda,
                 const uint16_t* b, int ldb, float beta,
                 float* c, int ldc,
                 const Level3Config& cfg, DispatchContext* ctx);

// SSYMM / DSYMM — V12.ELIDE + V12.TILE
// Lower/upper triangle only; C triangle computed from symmetric A.
VpuResult ssymm(char side, char uplo, int m, int n,
                float alpha, const float* a, int lda,
                const float* b, int ldb, float beta,
                float* c, int ldc,
                const Level3Config& cfg, DispatchContext* ctx);

VpuResult dsymm(char side, char uplo, int m, int n,
                double alpha, const double* a, int lda,
                const double* b, int ldb, double beta,
                double* c, int ldc,
                const Level3Config& cfg, DispatchContext* ctx);

// SSYRK / DSYRK — V12.ELIDE + V12.TILE + V12.FUSE
// C triangle only; fuse read + write of the updated triangle.
VpuResult ssyrk(char uplo, char trans, int n, int k,
                float alpha, const float* a, int lda,
                float beta, float* c, int ldc,
                const Level3Config& cfg, DispatchContext* ctx);

VpuResult dsyrk(char uplo, char trans, int n, int k,
                double alpha, const double* a, int lda,
                double beta, double* c, int ldc,
                const Level3Config& cfg, DispatchContext* ctx);

// SSYR2K / DSYR2K — V12.ELIDE + V12.TILE + V12.FUSE
// Two GEMMs fused; upper triangle only.
VpuResult ssyr2k(char uplo, char trans, int n, int k,
                 float alpha, const float* a, int lda,
                 const float* b, int ldb, float beta,
                 float* c, int ldc,
                 const Level3Config& cfg, DispatchContext* ctx);

VpuResult dsyr2k(char uplo, char trans, int n, int k,
                 double alpha, const double* a, int lda,
                 const double* b, int ldb, double beta,
                 double* c, int ldc,
                 const Level3Config& cfg, DispatchContext* ctx);

// STRMM / DTRMM — V12.ELIDE + V12.TILE
// Skip zero triangle; M×N tile.
VpuResult strmm(char side, char uplo, char transa, char diag,
                int m, int n, float alpha,
                const float* a, int lda,
                float* b, int ldb,
                const Level3Config& cfg, DispatchContext* ctx);

VpuResult dtrmm(char side, char uplo, char transa, char diag,
                int m, int n, double alpha,
                const double* a, int lda,
                double* b, int ldb,
                const Level3Config& cfg, DispatchContext* ctx);

// STRSM / DTRSM — V12.TILE + V12.COST-GATE
// Panel solve: CPU when M < 512; GPU panel-batch when M ≥ 512.
VpuResult strsm(char side, char uplo, char transa, char diag,
                int m, int n, float alpha,
                const float* a, int lda,
                float* b, int ldb,
                const Level3Config& cfg, DispatchContext* ctx);

VpuResult dtrsm(char side, char uplo, char transa, char diag,
                int m, int n, double alpha,
                const double* a, int lda,
                double* b, int ldb,
                const Level3Config& cfg, DispatchContext* ctx);

// SGEMM_BATCH — V12.FUSE
// Group identical shapes into one GEMM dispatch.
VpuResult sgemm_batch(int group_count, const int* group_size,
                      char transa, char transb,
                      const int* m, const int* n, const int* k,
                      const float* alpha,
                      const float* const* a_array, const int* lda,
                      const float* const* b_array, const int* ldb,
                      const float* beta,
                      float* const* c_array, const int* ldc,
                      const Level3Config& cfg, DispatchContext* ctx);

// SGEMM_BATCH_STRIDED — V12.FUSE + V12.TILE
// Batched via OpenVINO; tile per element.
VpuResult sgemm_batch_strided(char transa, char transb,
                              int m, int n, int k,
                              float alpha,
                              const float* a, int lda, int stride_a,
                              const float* b, int ldb, int stride_b,
                              float beta,
                              float* c, int ldc, int stride_c,
                              int batch_count,
                              const Level3Config& cfg, DispatchContext* ctx);

// ── CPU reference implementations (R4) ────────────────────────────────
// Used to validate GPU/OpenVINO output. Always correct, not optimised.
void cpu_sgemm_ref(char transa, char transb, int m, int n, int k,
                   float alpha, const float* a, int lda,
                   const float* b, int ldb, float beta,
                   float* c, int ldc);

void cpu_dgemm_ref(char transa, char transb, int m, int n, int k,
                   double alpha, const double* a, int lda,
                   const double* b, int ldb, double beta,
                   double* c, int ldc);

void cpu_ssyrk_ref(char uplo, char trans, int n, int k,
                   float alpha, const float* a, int lda,
                   float beta, float* c, int ldc);

void cpu_ssyr2k_ref(char uplo, char trans, int n, int k,
                    float alpha, const float* a, int lda,
                    const float* b, int ldb, float beta,
                    float* c, int ldc);

void cpu_strsm_ref(char side, char uplo, char transa, char diag,
                   int m, int n, float alpha,
                   const float* a, int lda,
                   float* b, int ldb);

void cpu_ssymm_ref(char side, char uplo, int m, int n,
                   float alpha, const float* a, int lda,
                   const float* b, int ldb, float beta,
                   float* c, int ldc);

void cpu_strmm_ref(char side, char uplo, char transa, char diag,
                   int m, int n, float alpha,
                   const float* a, int lda,
                   float* b, int ldb);

// ── Workload contract builder (R3) ────────────────────────────────────
WorkloadContract make_gemm_contract(const char* op, int m, int n, int k,
                                    DType dt_a, DType dt_b, DType dt_c, DType dt_acc,
                                    float alpha, float beta,
                                    int trans_a, int trans_b, int uplo,
                                    MethodSet methods, Backend backend,
                                    const Level3Config& cfg);

// ── Result helpers (R1, R5, R7, R8) ───────────────────────────────────
VpuResult build_result(const WorkloadContract& c,
                       const TimingTriple& t,
                       const CorrectnessResult& corr,
                       const EnvironmentMeta& e);

// R1: compute 95% CI on the speedup ratio and set verdict.
void finalize_ci(VpuResult& r);

// R4: validate GPU output against CPU reference within contract tolerance.
CorrectnessResult validate_output(const float* gpu_out, const float* cpu_ref,
                                  size_t n_elements, float tolerance);

// R8: collect environment metadata (device, driver, Vulkan, shader hashes).
EnvironmentMeta collect_env(const DispatchContext* ctx);

// R1: high-resolution timer (microseconds).
double now_us();

// R7: measure GPU-timestamp, host, and E2E timing for a dispatch batch.
TimingTriple measure_batch(DispatchContext* ctx,
                           const WorkloadContract& c,
                           const void* push_data, uint32_t push_size,
                           const std::vector<const void*>& buffers);

} // namespace Vpu
