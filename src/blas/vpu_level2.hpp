// VPU-CR-PROVENANCE
// Mechanism:  V12.TILE + V12.REDUCE + V12.ELIDE + V12.FUSE + V12.COST-GATE (Level 2 BLAS)
// Derived-from: Lawson et al. (1979) §5-6; Golub & Van Loan (2013) §3.2;
//   VPU_CELL_CONCEPTS_v0.1 §2 (SLM query), §5.3 (H-C placement), §5.5 (H-E claim),
//   §6 (H-G alignment), §5.4 (H-F budget); VPU_CLEANROOM_POLICY_v0.1 §5 (method catalogue)
// Spec entry: spec/blas.md#level2
// Author:     agent:deploy3-shardB
// Exposure:   none

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "vpu_blas.hpp"
#include "optimized_baseline.hpp"

namespace VPU {

// ============================================================
// H-G: Tile descriptor with alignment rejection
// VPU_CELL_CONCEPTS §5.3: misaligned DMA fails silently (arXiv:0710.2442 §2 failure mode 2)
// ============================================================
struct TileDesc {
    size_t tile_m;
    size_t tile_n;
    size_t tile_k;
    size_t alignment_bytes;

    bool ValidateAlignment(size_t base_offset_bytes, size_t element_bytes) const;
};

// ============================================================
// H-F: SLM budget declared vs runtime-queried limit
// VPU_CELL_CONCEPTS §2: SLM size varies by Xe generation; must query at runtime
// ============================================================
struct SLMBudget {
    size_t declared_bytes;
    size_t runtime_limit_bytes;

    bool can_fit() const { return declared_bytes <= runtime_limit_bytes; }
};

// ============================================================
// Banded matrix descriptor (GBMV, SBMV, TBMV, TBSV)
// ============================================================
struct BandDesc {
    size_t kl;
    size_t ku;
};

// ============================================================
// Packed format descriptor (SPMV, SSPR, TPMV, TPSV)
// ============================================================
struct PackedDesc {
    size_t N;
    bool upper;
};

// ============================================================
// H-G: Reject misaligned tile descriptors before dispatch
// ============================================================
inline bool CheckTileAlignment(const TileDesc& tile,
                               size_t base_offset_bytes,
                               size_t element_bytes) {
    return tile.ValidateAlignment(base_offset_bytes, element_bytes);
}

// ============================================================
// H-F: Query SLM budget from OpenVINO runtime context
// VPU_CELL_CONCEPTS §2: per-Xe-core SLM size must be queried, not assumed
// ============================================================
SLMBudget QuerySLMBudget(std::shared_ptr<Context> ctx,
                         size_t declared_bytes,
                         const std::string& device_name = "GPU");

// ============================================================
// R8: Environment record helpers
// ============================================================
std::string ComputeSourceHash(const std::string& source);
std::string ComputeShaderHash(const std::string& spirv);

// ============================================================
// CPU reference implementations (R4: correctness gate)
// Clean-room implementations from Lawson et al. (1979) §5-6,
//   Golub & Van Loan (2013) §3.2
// ============================================================
namespace Level2Ref {

// --- GEMV: y = alpha * op(A) * x + beta * y, A is M×N ---
void sgemv_ref(bool trans, size_t M, size_t N, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
void dgemv_ref(bool trans, size_t M, size_t N, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

// --- GER: A = alpha * x * y^T + A, A is M×N ---
void sger_ref(size_t M, size_t N, float alpha,
              const float* x, size_t incx,
              const float* y, size_t incy,
              float* A, size_t lda);
void dger_ref(size_t M, size_t N, double alpha,
              const double* x, size_t incx,
              const double* y, size_t incy,
              double* A, size_t lda);

// --- SYR: A = alpha * x * x^T + A (symmetric rank-1, upper triangle) ---
void ssyr_ref(size_t N, float alpha,
              const float* x, size_t incx,
              float* A, size_t lda);
void dsyr_ref(size_t N, double alpha,
              const double* x, size_t incx,
              double* A, size_t lda);

// --- SYR2: A = alpha * x * y^T + alpha * y * x^T + A (symmetric rank-2) ---
void ssyr2_ref(size_t N, float alpha,
               const float* x, size_t incx,
               const float* y, size_t incy,
               float* A, size_t lda);
void dsyr2_ref(size_t N, double alpha,
               const double* x, size_t incx,
               const double* y, size_t incy,
               double* A, size_t lda);

// --- SYMV: y = alpha * A * x + beta * y (symmetric matrix-vector) ---
void ssymv_ref(size_t N, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
void dsymv_ref(size_t N, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

// --- TRMV: x = op(A) * x (triangular, upper triangle) ---
void strmv_ref(bool trans, bool unit_diag,
               size_t N, const float* A, size_t lda,
               float* x, size_t incx);
void dtrmv_ref(bool trans, bool unit_diag,
               size_t N, const double* A, size_t lda,
               double* x, size_t incx);

// --- TRSV: solve A * x = b (triangular, upper triangle) ---
void strsv_ref(bool trans, bool unit_diag,
               size_t N, const float* A, size_t lda,
               float* x, size_t incx);
void dtrsv_ref(bool trans, bool unit_diag,
               size_t N, const double* A, size_t lda,
               double* x, size_t incx);

// --- GBMV: y = alpha * op(A) * x + beta * y (banded) ---
void sgbmv_ref(bool trans, size_t M, size_t N, size_t kl, size_t ku,
               float alpha, const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
void dgbmv_ref(bool trans, size_t M, size_t N, size_t kl, size_t ku,
               double alpha, const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

// --- SBMV: y = alpha * A * x + beta * y (symmetric banded) ---
void ssbmv_ref(size_t N, size_t k, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
void dsbmv_ref(size_t N, size_t k, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

// --- TBMV: x = op(A) * x (triangular banded) ---
void stbmv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const float* A, size_t lda,
               float* x, size_t incx);
void dtbmv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const double* A, size_t lda,
               double* x, size_t incx);

// --- TBSV: solve A * x = b (triangular banded) ---
void stbsv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const float* A, size_t lda,
               float* x, size_t incx);
void dtbsv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const double* A, size_t lda,
               double* x, size_t incx);

// --- SPMV: y = alpha * A * x + beta * y (packed symmetric) ---
void spmv_ref(size_t N, float alpha,
              const float* AP,
              const float* x, size_t incx,
              float beta, float* y, size_t incy);
void dspmv_ref(size_t N, double alpha,
               const double* AP,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

// --- SSPR: A = alpha * x * x^T + A (packed symmetric rank-1) ---
void sspr_ref(size_t N, float alpha,
              const float* x, size_t incx,
              float* AP);
void dspr_ref(size_t N, double alpha,
              const double* x, size_t incx,
              double* AP);

// --- TPMV: x = op(A) * x (packed triangular) ---
void stpmv_ref(bool trans, bool unit_diag,
               size_t N, const float* AP,
               float* x, size_t incx);
void dtpmv_ref(bool trans, bool unit_diag,
               size_t N, const double* AP,
               double* x, size_t incx);

// --- TPSV: solve A * x = b (packed triangular) ---
void stpsv_ref(bool trans, bool unit_diag,
               size_t N, const float* AP,
               float* x, size_t incx);
void dtpsv_ref(bool trans, bool unit_diag,
               size_t N, const double* AP,
               double* x, size_t incx);

} // namespace Level2Ref

// ============================================================
// Non-standard aliases (shardB.txt: spv, dpv, stri, dtri, dpmv)
// These are wrapper aliases for standard BLAS packed/solve functions
// ============================================================
// spv/dpv = spmv/dspmv (packed symmetric matrix-vector multiply)
// stri/dtri = strsv/dtrsv (inverted triangular solve)
// dpmv = dspmv (packed symmetric matrix-vector multiply, double)

// ============================================================
// Debug hypothesis tests (R1 compliance)
// VPU_CELL_CONCEPTS §7: each hypothesis has a kill condition
// ============================================================
namespace Level2Hypo {

struct HypothesisResult {
    std::string name;
    std::string decision;
    double speedup_ratio;
    double ci_lower;
    double ci_upper;
    bool passed;
};

// H-GEMV-TILE-1: SLM tile 256×64 saturates L2 bandwidth on Xe-LPG.
// Below 256 outputs, dispatch overhead dominates.
// Decision: if M < 512 → CPU; if M ≥ 512 → GPU with tile.
// Kill: if GPU <5% faster at M ≥ 512.
// H-GEMV-TILE-1
HypothesisResult H_GEMV_TILE_1(VPU_BLAS& blas, size_t M, size_t K);

// H-GEMV-TRANS-1: Transpose GEMV (y = A^T x) has no data sharing between
// output elements. Hypothesis: SLM tile does not help; GPU only wins when
// M × K > 16M elements.
// Kill: if CPU wins at M×K > 16M.
HypothesisResult H_GEMV_TRANS_1(VPU_BLAS& blas, size_t M, size_t K);

// H-SYMM-ELIDE-1: ssymv with V12.ELIDE computes N(N+1)/2 vs N² elements.
// Hypothesis: GPU wins only when the 2x work reduction + SLM tile
// overcomes dispatch overhead.
// Decision: triangle size > 4096² → GPU; else CPU.
// Kill: if elision saves < 10% at 4096².
HypothesisResult H_SYMM_ELIDE_1(VPU_BLAS& blas, size_t N);

// H-TRSV-SEQUENTIAL-1: Forward substitution is inherently sequential
// along the diagonal. Hypothesis: GPU panel-solve with V12.REDUCE
// for panel-level parallelism only beats CPU when panel count > 128.
// Kill: if M < 128×128 → force CPU.
HypothesisResult H_TRSV_SEQUENTIAL_1(VPU_BLAS& blas, size_t M);

// H-GEMV-TILE-1-ENHANCED: V12.GPU vs optimized OpenBLAS baseline
// Test shapes: 1024x1024, 2048x2048 (4x batch), 4096x4096
// Target: 20x speedup on compute-bound workloads
HypothesisResult H_GEMV_TILE_1_ENHANCED(VPU_BLAS& blas, size_t M, size_t K);

} // namespace Level2Hypo

} // namespace VPU
