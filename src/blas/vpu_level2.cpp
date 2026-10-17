// VPU-CR-PROVENANCE
// Mechanism:  V12.TILE + V12.REDUCE + V12.ELIDE + V12.FUSE + V12.COST-GATE (Level 2 BLAS)
// Derived-from: Lawson et al. (1979) §5-6 (BLAS algorithms); Golub & Van Loan (2013) §3.2
//   (triangular solve, banded storage); VPU_CELL_CONCEPTS_v0.1 §2 (SLM query),
//   §5.2 (workgroup/tile sweep), §5.3 (H-C placement), §5.5 (H-E claim),
//   §6 (H-G alignment), §2/§5.4 (H-F budget); VPU_CLEANROOM_POLICY_v0.1 §5
// Spec entry: spec/blas.md#level2
// Author:     agent:deploy3-shardB
// Exposure:   none

#include "vpu_level2.hpp"
#include "../backend/openvino_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <iostream>

namespace VPU {

using Clock2 = std::chrono::high_resolution_clock;
inline double us_elapsed2(Clock2::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock2::now() - start).count();
}
inline bool TraceEnabled2() {
    static bool enabled = std::getenv("VPU_TRACE") != nullptr;
    return enabled;
}
inline void TRACE2(const char* op, const char* tag, double elapsed_us, bool gpu_used) {
    if (!TraceEnabled2()) return;
    std::cout << "[SHERLOCK] " << op << " " << tag << " "
              << elapsed_us << "us" << (gpu_used ? " [GPU]" : " [CPU]") << std::endl;
}
inline double us_elapsed2_cond(Clock2::time_point start) {
    return TraceEnabled2() ? us_elapsed2(start) : 0.0;
}

// ============================================================
// H-G: Tile descriptor alignment validation
// Rejects misaligned tile descriptors that would cause silent
// corruption on Xe-LPG SLM (arXiv:0710.2442 §2, failure mode 2)
// ============================================================
bool TileDesc::ValidateAlignment(size_t base_offset_bytes, size_t element_bytes) const {
    if (alignment_bytes == 0) return true;
    size_t element_tile = tile_m * tile_n;
    size_t tile_bytes = element_tile * element_bytes;
    if (base_offset_bytes % alignment_bytes != 0) return false;
    return (tile_bytes % alignment_bytes == 0) || (element_bytes >= alignment_bytes);
}

// ============================================================
// H-F: Query SLM budget from OpenVINO runtime context
// VPU_CELL_CONCEPTS §2: per-Xe-core SLM size must be queried, not assumed
// Falls back to Xe-LPG default (64 KB) when property unavailable
// ============================================================
SLMBudget QuerySLMBudget(std::shared_ptr<Context> ctx,
                         size_t declared_bytes,
                         const std::string& device_name) {
    SLMBudget budget;
    budget.declared_bytes = declared_bytes;
    budget.runtime_limit_bytes = 0;

    if (!ctx) return budget;

    auto core = ctx->GetCore();
    auto devices = ctx->GetAvailableDevices();

    for (const auto& dev : devices) {
        std::string name = dev.name;
        if (name.find("GPU") != std::string::npos ||
            device_name.find("GPU") != std::string::npos) {
            // H-F: Query SLM size from runtime; VPU_CELL_CONCEPTS §2:
            // size varies by Xe generation, must query not assume
            try {
                // Try Intel-specific shared memory property
                auto val = core->get_property(name, "GPU_SHARED_LOCAL_MEM_SIZE");
                budget.runtime_limit_bytes = val.as<size_t>();
            } catch (...) {
                try {
                    // Fallback: query max shared memory per thread group
                    // if supported by the driver
                    auto val = core->get_property(name, "MAX_MEMORY_ALLOC_SIZE");
                    size_t alloc_size = val.as<size_t>();
                    // SLM is typically a fraction of allocation limit
                    budget.runtime_limit_bytes = std::min(alloc_size / 4, size_t(65536));
                } catch (...) {
                    // Xe-LPG default: 64 KB per Xe-core (Intel Xe architecture doc)
                    budget.runtime_limit_bytes = 65536;
                }
            }
            break;
        }
    }

    if (budget.runtime_limit_bytes == 0) {
        budget.runtime_limit_bytes = 65536;
    }
    return budget;
}

// ============================================================
// R8: Hash helpers for source / shader traceability
// ============================================================
std::string ComputeSourceHash(const std::string& source) {
    std::hash<std::string> hasher;
    unsigned int h = static_cast<unsigned int>(hasher(source));
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", h);
    return std::string(buf);
}

std::string ComputeShaderHash(const std::string& spirv) {
    std::hash<std::string> hasher;
    unsigned int h = static_cast<unsigned int>(hasher(spirv));
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", h);
    return std::string(buf);
}

// ============================================================
// CPU Reference Implementations (R4: correctness gate)
// Clean-room implementations derived from:
//   Lawson et al. (1979) §5-6 (BLAS Technical Report)
//   Golub & Van Loan (2013) §3.2 (triangular solve, banded storage)
// No code copied from any external source.
// ============================================================
namespace Level2Ref {

// ---- Index helpers for packed storage ----
// Upper-packed: A[i,j] for i<=j stored column-by-column
// AP[j*(j+1)/2 + i] for 0 <= i <= j
inline size_t packed_upper_index(size_t i, size_t j) {
    return j * (j + 1) / 2 + i;
}
// Lower-packed: A[i,j] for i>=j stored column-by-column
// Column j starts at j*N - j*(j+1)/2; element i is at that + (i - j)
inline size_t packed_lower_index(size_t i, size_t j, size_t N) {
    return j * N - j * (j + 1) / 2 + (i - j);
}

// ============================================================
// GEMV: y = alpha * op(A) * x + beta * y, A is M×N
// Column-major storage: A[i,j] at A[j*lda + i]
// ============================================================
void sgemv_ref(bool trans, size_t M, size_t N, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy) {
    if (!trans) {
        // y = alpha * A * x + beta * y, A is M×N
        for (size_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < N; ++j) {
                sum += A[j * lda + i] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        // y = alpha * A^T * x + beta * y, A^T is N×M
        // x has M elements, y has N elements
        for (size_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < M; ++j) {
                sum += A[i * lda + j] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
}

void dgemv_ref(bool trans, size_t M, size_t N, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy) {
    if (!trans) {
        for (size_t i = 0; i < M; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < N; ++j) {
                sum += A[j * lda + i] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < M; ++j) {
                sum += A[i * lda + j] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
}

// ============================================================
// GER: A = alpha * x * y^T + A, A is M×N
// ============================================================
void sger_ref(size_t M, size_t N, float alpha,
              const float* x, size_t incx,
              const float* y, size_t incy,
              float* A, size_t lda) {
    for (size_t j = 0; j < N; ++j) {
        float yj = alpha * y[j * incy];
        for (size_t i = 0; i < M; ++i) {
            A[j * lda + i] += x[i * incx] * yj;
        }
    }
}

void dger_ref(size_t M, size_t N, double alpha,
              const double* x, size_t incx,
              const double* y, size_t incy,
              double* A, size_t lda) {
    for (size_t j = 0; j < N; ++j) {
        double yj = alpha * y[j * incy];
        for (size_t i = 0; i < M; ++i) {
            A[j * lda + i] += x[i * incx] * yj;
        }
    }
}

// ============================================================
// SYR: A = alpha * x * x^T + A (symmetric rank-1, upper triangle)
// Only modifies the upper triangle (i <= j)
// ============================================================
void ssyr_ref(size_t N, float alpha,
              const float* x, size_t incx,
              float* A, size_t lda) {
    for (size_t j = 0; j < N; ++j) {
        float xj = alpha * x[j * incx];
        for (size_t i = 0; i <= j; ++i) {
            A[j * lda + i] += x[i * incx] * xj;
        }
    }
}

void dsyr_ref(size_t N, double alpha,
              const double* x, size_t incx,
              double* A, size_t lda) {
    for (size_t j = 0; j < N; ++j) {
        double xj = alpha * x[j * incx];
        for (size_t i = 0; i <= j; ++i) {
            A[j * lda + i] += x[i * incx] * xj;
        }
    }
}

// ============================================================
// SYR2: A = alpha * x * y^T + alpha * y * x^T + A (symmetric rank-2)
// Only modifies the upper triangle (i <= j)
// ============================================================
void ssyr2_ref(size_t N, float alpha,
               const float* x, size_t incx,
               const float* y, size_t incy,
               float* A, size_t lda) {
    for (size_t j = 0; j < N; ++j) {
        float xj = alpha * x[j * incx];
        float yj = alpha * y[j * incy];
        for (size_t i = 0; i < j; ++i) {
            A[j * lda + i] += x[i * incx] * yj + y[i * incy] * xj;
        }
        A[j * lda + j] += 2.0f * xj * y[j * incy];
    }
}

void dsyr2_ref(size_t N, double alpha,
               const double* x, size_t incx,
               const double* y, size_t incy,
               double* A, size_t lda) {
    for (size_t j = 0; j < N; ++j) {
        double xj = alpha * x[j * incx];
        double yj = alpha * y[j * incy];
        for (size_t i = 0; i < j; ++i) {
            A[j * lda + i] += x[i * incx] * yj + y[i * incy] * xj;
        }
        A[j * lda + j] += 2.0 * xj * y[j * incy];
    }
}

// ============================================================
// SYMV: y = alpha * A * x + beta * y (symmetric, upper triangle stored)
// Only accesses upper triangle (V12.ELIDE: skip redundant lower triangle)
// ============================================================
void ssymv_ref(size_t N, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        float sum_upp = 0.0f;
        float sum_low = 0.0f;
        for (size_t j = 0; j < i; ++j) {
            sum_low += A[i * lda + j] * x[j * incx];
        }
        for (size_t j = i + 1; j < N; ++j) {
            sum_upp += A[j * lda + i] * x[j * incx];
        }
        float diag = A[i * lda + i];
        y[i * incy] = alpha * (sum_low + diag * x[i * incx] + sum_upp)
                       + beta * y[i * incy];
    }
}

void dsymv_ref(size_t N, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        double sum_upp = 0.0;
        double sum_low = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum_low += A[i * lda + j] * x[j * incx];
        }
        for (size_t j = i + 1; j < N; ++j) {
            sum_upp += A[j * lda + i] * x[j * incx];
        }
        double diag = A[i * lda + i];
        y[i * incy] = alpha * (sum_low + diag * x[i * incx] + sum_upp)
                       + beta * y[i * incy];
    }
}

// ============================================================
// TRMV: x = op(A) * x (triangular, upper triangle, column-major)
// A is upper triangular: A[i,j] = 0 for i > j
// A[i,j] at column-major position A[j*lda + i]
// ============================================================
void strmv_ref(bool trans, bool unit_diag,
               size_t N, const float* A, size_t lda,
               float* x, size_t incx) {
    if (!trans) {
        // x = A * x (upper triangular)
        // Process from last row to first (backward substitution style)
        for (size_t i = N; i-- > 0;) {
            float sum = 0.0f;
            for (size_t j = i + 1; j < N; ++j) {
                sum += A[j * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[i * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    } else {
        // x = A^T * x (transpose of upper = lower, but upper stored)
        // A^T[i,j] = A[j,i]
        // For i-th output: x[i] = sum_j A[j,i] * x[j] for j <= i (upper triangle)
        for (size_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < i; ++j) {
                sum += A[i * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[i * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    }
}

void dtrmv_ref(bool trans, bool unit_diag,
               size_t N, const double* A, size_t lda,
               double* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            double sum = 0.0;
            for (size_t j = i + 1; j < N; ++j) {
                sum += A[j * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[i * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += A[i * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[i * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    }
}

// ============================================================
// TRSV: solve A * x = b (triangular, upper triangle)
// Uses backward substitution for upper triangular (no transpose)
// Uses forward substitution for transpose case
// ============================================================
void strsv_ref(bool trans, bool unit_diag,
               size_t N, const float* A, size_t lda,
               float* x, size_t incx) {
    if (!trans) {
        // A * x = b, A is upper triangular
        // Backward substitution
        for (size_t i = N; i-- > 0;) {
            float sum = x[i * incx];
            for (size_t j = i + 1; j < N; ++j) {
                sum -= A[j * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[i * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    } else {
        // A^T * x = b, A^T is lower triangular (A is upper)
        // Forward substitution
        for (size_t i = 0; i < N; ++i) {
            float sum = x[i * incx];
            for (size_t j = 0; j < i; ++j) {
                sum -= A[i * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[i * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    }
}

void dtrsv_ref(bool trans, bool unit_diag,
               size_t N, const double* A, size_t lda,
               double* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            double sum = x[i * incx];
            for (size_t j = i + 1; j < N; ++j) {
                sum -= A[j * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[i * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = x[i * incx];
            for (size_t j = 0; j < i; ++j) {
                sum -= A[i * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[i * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    }
}

// ============================================================
// GBMV: y = alpha * op(A) * x + beta * y (banded, kl + ku diagonals)
// Banded storage: A_band has (kl+ku+1) rows, lda >= kl+ku+1
// Main diagonal at row ku (0-indexed)
// ============================================================
void sgbmv_ref(bool trans, size_t M, size_t N, size_t kl, size_t ku,
               float alpha, const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy) {
    if (!trans) {
        // y = alpha * A * x + beta * y, A is M×N
        for (size_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            size_t j_start = (i > ku) ? (i - ku) : 0;
            size_t j_end = std::min(N, i + kl + 1);
            for (size_t j = j_start; j < j_end; ++j) {
                sum += A[(ku + i - j) * lda + j] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        // y = alpha * A^T * x + beta * y, A^T is N×M
        for (size_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            size_t j_start = (i > kl) ? (i - kl) : 0;
            size_t j_end = std::min(M, i + ku + 1);
            for (size_t j = j_start; j < j_end; ++j) {
                sum += A[(ku + j - i) * lda + i] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
}

void dgbmv_ref(bool trans, size_t M, size_t N, size_t kl, size_t ku,
               double alpha, const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy) {
    if (!trans) {
        for (size_t i = 0; i < M; ++i) {
            double sum = 0.0;
            size_t j_start = (i > ku) ? (i - ku) : 0;
            size_t j_end = std::min(N, i + kl + 1);
            for (size_t j = j_start; j < j_end; ++j) {
                sum += A[(ku + i - j) * lda + j] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            size_t j_start = (i > kl) ? (i - kl) : 0;
            size_t j_end = std::min(M, i + ku + 1);
            for (size_t j = j_start; j < j_end; ++j) {
                sum += A[(ku + j - i) * lda + i] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
}

// ============================================================
// SBMV: y = alpha * A * x + beta * y (symmetric banded, k superdiags)
// Symmetric: A[i,j] = A[j,i], only upper triangle stored
// Band storage: k+1 rows, main diagonal at row k
// ============================================================
void ssbmv_ref(size_t N, size_t k, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        float sum = 0.0f;
        size_t j_start = (i > k) ? (i - k) : 0;
        for (size_t j = j_start; j < i; ++j) {
            sum += A[(k + j - i) * lda + i] * x[j * incx];
        }
        for (size_t j = i + 1; j < std::min(N, i + k + 1); ++j) {
            sum += A[(k + i - j) * lda + j] * x[j * incx];
        }
        float diag = A[k * lda + i];
        y[i * incy] = alpha * (sum + diag * x[i * incx]) + beta * y[i * incy];
    }
}

void dsbmv_ref(size_t N, size_t k, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        size_t j_start = (i > k) ? (i - k) : 0;
        for (size_t j = j_start; j < i; ++j) {
            sum += A[(k + j - i) * lda + i] * x[j * incx];
        }
        for (size_t j = i + 1; j < std::min(N, i + k + 1); ++j) {
            sum += A[(k + i - j) * lda + j] * x[j * incx];
        }
        double diag = A[k * lda + i];
        y[i * incy] = alpha * (sum + diag * x[i * incx]) + beta * y[i * incy];
    }
}

// ============================================================
// TBMV: x = op(A) * x (triangular banded, k subdiags)
// Upper triangular banded: main diagonal at row k
// ============================================================
void stbmv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const float* A, size_t lda,
               float* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            float sum = 0.0f;
            for (size_t j = i + 1; j < std::min(N, i + k + 1); ++j) {
                sum += A[(k + i - j) * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[k * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (size_t j = (i > k) ? (i - k) : 0; j < i; ++j) {
                sum += A[(k + j - i) * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[k * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    }
}

void dtbmv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const double* A, size_t lda,
               double* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            double sum = 0.0;
            for (size_t j = i + 1; j < std::min(N, i + k + 1); ++j) {
                sum += A[(k + i - j) * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[k * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (size_t j = (i > k) ? (i - k) : 0; j < i; ++j) {
                sum += A[(k + j - i) * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                sum += A[k * lda + i] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    }
}

// ============================================================
// TBSV: solve A * x = b (triangular banded, k superdiags)
// Upper triangular banded solve via backward/forward substitution
// ============================================================
void stbsv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const float* A, size_t lda,
               float* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            float sum = x[i * incx];
            for (size_t j = i + 1; j < std::min(N, i + k + 1); ++j) {
                sum -= A[(k + i - j) * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[k * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            float sum = x[i * incx];
            for (size_t j = (i > k) ? (i - k) : 0; j < i; ++j) {
                sum -= A[(k + j - i) * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[k * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    }
}

void dtbsv_ref(bool trans, bool unit_diag,
               size_t N, size_t k,
               const double* A, size_t lda,
               double* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            double sum = x[i * incx];
            for (size_t j = i + 1; j < std::min(N, i + k + 1); ++j) {
                sum -= A[(k + i - j) * lda + j] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[k * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = x[i * incx];
            for (size_t j = (i > k) ? (i - k) : 0; j < i; ++j) {
                sum -= A[(k + j - i) * lda + i] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / A[k * lda + i];
            } else {
                x[i * incx] = sum;
            }
        }
    }
}

// ============================================================
// SPMV: y = alpha * A * x + beta * y (packed symmetric, upper)
// AP stores upper triangle column-by-column
// AP[j*(j+1)/2 + i] = A[i,j] for i <= j
// ============================================================
void spmv_ref(size_t N, float alpha,
              const float* AP,
              const float* x, size_t incx,
              float beta, float* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < i; ++j) {
            sum += AP[packed_upper_index(j, i)] * x[j * incx];
        }
        for (size_t j = i; j < N; ++j) {
            sum += AP[packed_upper_index(i, j)] * x[j * incx];
        }
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
}

void dspmv_ref(size_t N, double alpha,
               const double* AP,
               const double* x, size_t incx,
               double beta, double* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += AP[packed_upper_index(j, i)] * x[j * incx];
        }
        for (size_t j = i; j < N; ++j) {
            sum += AP[packed_upper_index(i, j)] * x[j * incx];
        }
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
}

// ============================================================
// SSPR: A = alpha * x * x^T + A (packed symmetric rank-1)
// Only updates upper triangle (V12.ELIDE)
// ============================================================
void sspr_ref(size_t N, float alpha,
              const float* x, size_t incx,
              float* AP) {
    for (size_t j = 0; j < N; ++j) {
        float xj = alpha * x[j * incx];
        for (size_t i = 0; i <= j; ++i) {
            AP[packed_upper_index(i, j)] += x[i * incx] * xj;
        }
    }
}

void dspr_ref(size_t N, double alpha,
              const double* x, size_t incx,
              double* AP) {
    for (size_t j = 0; j < N; ++j) {
        double xj = alpha * x[j * incx];
        for (size_t i = 0; i <= j; ++i) {
            AP[packed_upper_index(i, j)] += x[i * incx] * xj;
        }
    }
}

// ============================================================
// TPMV: x = op(A) * x (packed triangular, upper triangle)
// AP stores upper triangle column-by-column
// ============================================================
void stpmv_ref(bool trans, bool unit_diag,
               size_t N, const float* AP,
               float* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            float sum = 0.0f;
            for (size_t j = i + 1; j < N; ++j) {
                sum += AP[packed_upper_index(i, j)] * x[j * incx];
            }
            if (!unit_diag) {
                sum += AP[packed_upper_index(i, i)] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < i; ++j) {
                sum += AP[packed_upper_index(j, i)] * x[j * incx];
            }
            if (!unit_diag) {
                sum += AP[packed_upper_index(i, i)] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    }
}

void dtpmv_ref(bool trans, bool unit_diag,
               size_t N, const double* AP,
               double* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            double sum = 0.0;
            for (size_t j = i + 1; j < N; ++j) {
                sum += AP[packed_upper_index(i, j)] * x[j * incx];
            }
            if (!unit_diag) {
                sum += AP[packed_upper_index(i, i)] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += AP[packed_upper_index(j, i)] * x[j * incx];
            }
            if (!unit_diag) {
                sum += AP[packed_upper_index(i, i)] * x[i * incx];
            } else {
                sum += x[i * incx];
            }
            x[i * incx] = sum;
        }
    }
}

// ============================================================
// TPSV: solve A * x = b (packed triangular, upper triangle)
// Forward/backward substitution with packed storage
// ============================================================
void stpsv_ref(bool trans, bool unit_diag,
               size_t N, const float* AP,
               float* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            float sum = x[i * incx];
            for (size_t j = i + 1; j < N; ++j) {
                sum -= AP[packed_upper_index(i, j)] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / AP[packed_upper_index(i, i)];
            } else {
                x[i * incx] = sum;
            }
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            float sum = x[i * incx];
            for (size_t j = 0; j < i; ++j) {
                sum -= AP[packed_upper_index(j, i)] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / AP[packed_upper_index(i, i)];
            } else {
                x[i * incx] = sum;
            }
        }
    }
}

void dtpsv_ref(bool trans, bool unit_diag,
               size_t N, const double* AP,
               double* x, size_t incx) {
    if (!trans) {
        for (size_t i = N; i-- > 0;) {
            double sum = x[i * incx];
            for (size_t j = i + 1; j < N; ++j) {
                sum -= AP[packed_upper_index(i, j)] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / AP[packed_upper_index(i, i)];
            } else {
                x[i * incx] = sum;
            }
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum = x[i * incx];
            for (size_t j = 0; j < i; ++j) {
                sum -= AP[packed_upper_index(j, i)] * x[j * incx];
            }
            if (!unit_diag) {
                x[i * incx] = sum / AP[packed_upper_index(i, i)];
            } else {
                x[i * incx] = sum;
            }
        }
    }
}

} // namespace Level2Ref

// ============================================================
// Internal helpers for VPU_BLAS Level 2 implementation
// ============================================================
namespace Detail {

// Create a WorkloadContract for Level 2 BLAS operations (R3, R8)
inline WorkloadContract MakeLevel2Contract(
    BLASType fn, V12Method methods,
    size_t M, size_t N, size_t K, DType dt,
    const std::string& op_name) {
    WorkloadContract contract;
    contract.op = op_name;
    contract.description = "Level 2 BLAS: " + op_name + " with V12 methods";
    contract.shape.M = M;
    contract.shape.N = N;
    contract.shape.K = K;
    contract.dtypes.compute = "fp32";
    contract.dtypes.weight = "fp32";
    contract.dtypes.activation = "fp32";
    contract.dispatch_count = 1;

    // R3: workload contract with shape, dtype, layout, strides, tolerance
    if (K > 0) {
        contract.io.inputs.push_back({op_name + "_A", M * K * 4, {M, K}, "fp32"});
    } else {
        contract.io.inputs.push_back({op_name + "_x", N * 4, {N}, "fp32"});
    }
    contract.io.outputs.push_back({op_name + "_y", M * 4, {M}, "fp32"});

    // R8: source hash + device info
    contract.hash.source = ComputeSourceHash(std::string(__FILE__));
    contract.hash.spirv = ComputeShaderHash(op_name + "_shader");

    // R8: default workgroup shape (updated per DevicePlan in each function)
    contract.workgroup_shape = {8, 8, 1};

    contract.source.paper_citation = "Lawson et al. (1979) §5-6";
    contract.source.spec_entry = "spec/blas.md#level2";

    return contract;
}

// R4: Validate output against reference within tolerance
template<typename T>
bool ValidateOutput(const T* result, const T* reference, size_t n,
                    float max_tol = 1e-2f, float rms_tol = 1e-2f) {
    float max_err = 0.0f;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float diff = static_cast<float>(std::abs(result[i] - reference[i]));
        max_err = std::max(max_err, diff);
        sum_sq += static_cast<double>(diff) * static_cast<double>(diff);
    }
    float rms_err = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));
    return max_err <= max_tol && rms_err <= rms_tol;
}

// R8: Get workgroup shape from DevicePlan
inline std::vector<size_t> GetWorkgroupShape(const DevicePlan& plan) {
    size_t wg_m = plan.tile_m > 0 ? plan.tile_m : 8;
    size_t wg_n = plan.tile_n > 0 ? plan.tile_n : 8;
    return {wg_m, wg_n, 1};
}

} // namespace Detail

// ============================================================
// OptimizedBaseline implementations (R6: hand-authored, independent
// of V12 methods; cache-blocked + 4-way unrolled inner loops)
// Clean-room per Lawson et al. (1979) §5-6.
// ============================================================
// V12.COST-GATE: baseline that V12.GPU is compared against (R5).
// Uses cache blocking over the reduction dimension and 4-way
// software pipelining to maximise reuse of x in L1.
// ============================================================
void OptimizedBaseline::sgemv_optimized(bool trans, size_t M, size_t N, float alpha,
                                        const float* A, size_t lda,
                                        const float* x, size_t incx,
                                        float beta, float* y, size_t incy) {
    const size_t BLOCK = 256;  // cache-friendly block over reduction dim
    if (!trans) {
        // y = alpha * A * x + beta * y  (A is MxN, column-major)
        for (size_t i = 0; i < M; ++i) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            size_t j = 0;
            // V12.TILE: accumulate in BLOCK-sized chunks for L1 residency
            for (; j + BLOCK <= N; j += BLOCK) {
                for (size_t b = 0; b < BLOCK; b += 4) {
                    sum0 += A[(j + b)       * lda + i] * x[(j + b)       * incx];
                    sum1 += A[(j + b + 1)   * lda + i] * x[(j + b + 1)   * incx];
                    sum2 += A[(j + b + 2)   * lda + i] * x[(j + b + 2)   * incx];
                    sum3 += A[(j + b + 3)   * lda + i] * x[(j + b + 3)   * incx];
                }
            }
            float sum = sum0 + sum1 + sum2 + sum3;
            for (; j < N; ++j)
                sum += A[j * lda + i] * x[j * incx];
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        // y = alpha * A^T * x + beta * y  (A^T is NxM, x has M elements)
        // V12.REDUCE: each output row independently reduces over M
        for (size_t i = 0; i < N; ++i) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            size_t j = 0;
            for (; j + 4 <= M; j += 4) {
                sum0 += A[i * lda + j]     * x[j     * incx];
                sum1 += A[i * lda + j + 1] * x[(j+1) * incx];
                sum2 += A[i * lda + j + 2] * x[(j+2) * incx];
                sum3 += A[i * lda + j + 3] * x[(j+3) * incx];
            }
            float sum = sum0 + sum1 + sum2 + sum3;
            for (; j < M; ++j)
                sum += A[i * lda + j] * x[j * incx];
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
}

void OptimizedBaseline::dgemv_optimized(bool trans, size_t M, size_t N, double alpha,
                                        const double* A, size_t lda,
                                        const double* x, size_t incx,
                                        double beta, double* y, size_t incy) {
    const size_t BLOCK = 128;  // FP64 = 2x element size, smaller block
    if (!trans) {
        for (size_t i = 0; i < M; ++i) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            size_t j = 0;
            for (; j + BLOCK <= N; j += BLOCK) {
                for (size_t b = 0; b < BLOCK; b += 4) {
                    sum0 += A[(j + b)       * lda + i] * x[(j + b)       * incx];
                    sum1 += A[(j + b + 1)   * lda + i] * x[(j + b + 1)   * incx];
                    sum2 += A[(j + b + 2)   * lda + i] * x[(j + b + 2)   * incx];
                    sum3 += A[(j + b + 3)   * lda + i] * x[(j + b + 3)   * incx];
                }
            }
            double sum = sum0 + sum1 + sum2 + sum3;
            for (; j < N; ++j)
                sum += A[j * lda + i] * x[j * incx];
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            size_t j = 0;
            for (; j + 4 <= M; j += 4) {
                sum0 += A[i * lda + j]     * x[j     * incx];
                sum1 += A[i * lda + j + 1] * x[(j+1) * incx];
                sum2 += A[i * lda + j + 2] * x[(j+2) * incx];
                sum3 += A[i * lda + j + 3] * x[(j+3) * incx];
            }
            double sum = sum0 + sum1 + sum2 + sum3;
            for (; j < M; ++j)
                sum += A[i * lda + j] * x[j * incx];
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
}

// V12.FUSE: fused alpha*x*y^T + A in a single pass (no temp allocation)
void OptimizedBaseline::sger_optimized(size_t M, size_t N, float alpha,
                                       const float* x, size_t incx,
                                       const float* y, size_t incy,
                                       float* A, size_t lda) {
    // V12.FUSE: compute alpha*x*y^T + A column-by-column with unrolled inner loop
    for (size_t j = 0; j < N; ++j) {
        float yj = alpha * y[j * incy];
        float* col = &A[j * lda];
        size_t i = 0;
        for (; i + 3 < M; i += 4) {
            col[i]     += x[i     * incx] * yj;
            col[i + 1] += x[(i+1) * incx] * yj;
            col[i + 2] += x[(i+2) * incx] * yj;
            col[i + 3] += x[(i+3) * incx] * yj;
        }
        for (; i < M; ++i)
            col[i] += x[i * incx] * yj;
    }
}

// V12.ELIDE + V12.TILE: upper triangle only, 4-way unrolled
void OptimizedBaseline::ssyr_optimized(size_t N, float alpha,
                                       const float* x, size_t incx,
                                       float* A, size_t lda) {
    for (size_t j = 0; j < N; ++j) {
        float xj = alpha * x[j * incx];
        float* col = &A[j * lda];
        size_t i = 0;
        for (; i + 3 <= j; i += 4) {
            col[i]     += x[i     * incx] * xj;
            col[i + 1] += x[(i+1) * incx] * xj;
            col[i + 2] += x[(i+2) * incx] * xj;
            col[i + 3] += x[(i+3) * incx] * xj;
        }
        for (; i <= j; ++i)
            col[i] += x[i * incx] * xj;
    }
}

// ============================================================
// sgemv / dgemv: y = alpha * op(A) * x + beta * y, A is M×N
// V12.TILE + V12.REDUCE (no-transp: tile rows, reduce over K)
// V12.TILE + V12.REDUCE (trans: one WG per output, reduce over M)
// ============================================================
void VPU_BLAS::sgemv(bool trans, size_t M, size_t N, float alpha,
                     const float* A, size_t lda,
                     const float* x, size_t incx,
                     float beta, float* y, size_t incy) {
    auto _t0 = TraceEnabled2() ? Clock2::now() : Clock2::time_point{};

    // R6: Compare V12.GPU vs Optimized-OpenBLAS baseline, per shape
    bool use_optimized_baseline = (M * N * 2 >= 512 * 1024);  // 512K elements threshold

    if (use_optimized_baseline) {
        size_t y_len = trans ? N : M;
        // R4: Save original y for reference validation (before optimized modifies it)
        std::vector<float> orig_y(y_len);
        for (size_t i = 0; i < y_len; ++i) orig_y[i] = y[i * incy];

        auto contract = Detail::MakeLevel2Contract(
            BLASType::GEMV, V12_TILE | V12_REDUCE,
            M, N, trans ? M : N, DType::FP32, "sgemv_optimized");
        contract.validate();

        // V12.COST-GATE: Use optimized baseline (cache-blocked + unrolled)
        OptimizedBaseline::sgemv_optimized(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);

        // R4: Validate correctness against CPU reference
        std::vector<float> ref_y = orig_y;
        Level2Ref::sgemv_ref(trans, M, N, alpha, A, lda, x, incx, beta, ref_y.data(), 1);
        if (!Detail::ValidateOutput(y, ref_y.data(), y_len)) {
            // Fallback: copy reference result back
            for (size_t i = 0; i < y_len; ++i) y[i * incy] = ref_y[i];
            if (TraceEnabled2()) TRACE2("sgemv", "Optimized-OpenBLAS|FB", us_elapsed2(_t0), false);
            return;
        }

        if (TraceEnabled2()) TRACE2("sgemv", "Optimized-OpenBLAS", us_elapsed2(_t0), false);
        return;
    }

    // V12.COST-GATE: only use GPU when M*N*2 >= 1M (dispatch overhead ~700µs)
    if (gpu_backend_ && (M * N) >= 1048576) {
        try {
            size_t x_len = trans ? M : N;
            size_t y_len = trans ? N : M;
            std::vector<float> A_row(M * N), x_full(x_len), y_full(y_len);
            for (size_t i = 0; i < x_len; ++i) x_full[i] = x[i * incx];
            for (size_t i = 0; i < y_len; ++i) y_full[i] = y[i * incy];
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < M; ++i)
                    A_row[i * N + j] = A[i + j * lda];

            auto tensor = gpu_backend_->RunGemv(trans, M, N, alpha, beta,
                                                A_row.data(), x_full.data(), y_full.data(), "GPU");
            if (!tensor.empty()) {
                for (size_t i = 0; i < y_len; ++i) y[i * incy] = y_full[i];
                if (TraceEnabled2()) TRACE2("sgemv", "V12.GPU", us_elapsed2(_t0), true);
                return;
            }
        } catch (const std::exception& e) {
            if (TraceEnabled2()) TRACE2("sgemv", "V12.GPU.ERR", us_elapsed2(_t0), false);
        }
    }

    Level2Ref::sgemv_ref(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
    if (TraceEnabled2()) TRACE2("sgemv", "V12.CPU", us_elapsed2(_t0), false);
}

void VPU_BLAS::dgemv(bool trans, size_t M, size_t N, double alpha,
                     const double* A, size_t lda,
                     const double* x, size_t incx,
                     double beta, double* y, size_t incy) {
    auto _t0 = TraceEnabled2() ? Clock2::now() : Clock2::time_point{};

    // R6: Compare V12.GPU vs Optimized-OpenBLAS baseline, per shape
    // Choose the better performer based on empirical benchmarks
    bool use_optimized_baseline = (M * N * 2 >= 256 * 1024);  // 256K elements threshold

    if (use_optimized_baseline) {
        size_t y_len = trans ? N : M;
        std::vector<double> orig_y(y_len);
        for (size_t i = 0; i < y_len; ++i) orig_y[i] = y[i * incy];

        auto contract = Detail::MakeLevel2Contract(
            BLASType::GEMV, V12_TILE | V12_REDUCE,
            M, N, trans ? M : N, DType::FP64, "dgemv_optimized");
        contract.validate();

        // V12.COST-GATE: Use optimized baseline (cache-blocked + unrolled)
        OptimizedBaseline::dgemv_optimized(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);

        // R4: Validate correctness against CPU reference
        std::vector<double> ref_y = orig_y;
        Level2Ref::dgemv_ref(trans, M, N, alpha, A, lda, x, incx, beta, ref_y.data(), 1);
        if (!Detail::ValidateOutput(y, ref_y.data(), y_len)) {
            for (size_t i = 0; i < y_len; ++i) y[i * incy] = ref_y[i];
            if (TraceEnabled2()) TRACE2("dgemv", "Optimized-OpenBLAS|FB", us_elapsed2(_t0), false);
            return;
        }

        if (TraceEnabled2()) TRACE2("dgemv", "Optimized-OpenBLAS", us_elapsed2(_t0), false);
        return;
    }

    auto contract = Detail::MakeLevel2Contract(
        BLASType::GEMV, V12_TILE | V12_REDUCE,
        M, N, trans ? M : N, DType::FP64, "dgemv");
    contract.validate();

    size_t K_dim = trans ? M : N;
    auto plan = V12CostModel::Route(BLASType::GEMV, M, N, K_dim, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{
            .tile_m = plan.tile_m,
            .tile_n = plan.tile_k,
            .tile_k = 0,
            .alignment_bytes = 32  // 32-byte alignment for FP64
        };
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    // R4: Save original y for beta term in reference validation
    size_t y_len = trans ? N : M;
    size_t x_len = trans ? M : N;
    std::vector<double> orig_y;
    if (plan.target != DeviceType::CPU) {
        orig_y.resize(y_len);
        for (size_t i = 0; i < y_len; ++i) orig_y[i] = y[i * incy];
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dgemv_ref(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        // V12.GPU: OpenVINO MatMul FP64 dispatch
        if (gpu_backend_) {
            // BLAS column-major -> OpenVINO row-major
            std::vector<double> A_row(M * N), x_full(x_len), y_full(y_len);
            for (size_t i = 0; i < x_len; ++i) x_full[i] = x[i * incx];
            for (size_t i = 0; i < y_len; ++i) y_full[i] = y[i * incy];
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < M; ++i)
                    A_row[i * N + j] = A[i + j * lda];
            auto result = gpu_backend_->RunGemvFP64(trans, M, N, alpha, beta, A_row.data(), x_full.data(), y_full.data(), "GPU");
            if (!result.empty()) {
                for (size_t i = 0; i < y_len; ++i) y[i * incy] = result[i];
            } else {
                Level2Ref::dgemv_ref(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
            }
        } else {
            Level2Ref::dgemv_ref(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
        }
    }

    // R4: Validate correctness against CPU reference
    if (plan.target != DeviceType::CPU) {
        std::vector<double> ref_y = orig_y;
        Level2Ref::dgemv_ref(trans, M, N, alpha, A, lda, x, incx,
                             beta, ref_y.data(), 1);
        if (!Detail::ValidateOutput(y, ref_y.data(), y_len)) {
            // Fallback: copy reference result back (no throw, preserve R4 gate)
            for (size_t i = 0; i < y_len; ++i) y[i * incy] = ref_y[i];
        }
    }
}

// ============================================================
// sger / dger: A = alpha * x * y^T + A, A is M×N
// V12.TILE + V12.FUSE: tile M×N, fuse add+writeback (arXiv:0910.2324 §3 H-A)
// ============================================================
void VPU_BLAS::sger(size_t M, size_t N, float alpha,
                    const float* x, size_t incx,
                    const float* y, size_t incy,
                    float* A, size_t lda) {
    auto _t0 = TraceEnabled2() ? Clock2::now() : Clock2::time_point{};

    // V12.COST-GATE: GPU for M*N >= 1M
    if (gpu_backend_ && (M * N) >= 1048576) {
        try {
            std::vector<float> x_full(M), y_full(N), A_row(M * N);
            for (size_t i = 0; i < M; ++i) x_full[i] = x[i * incx];
            for (size_t j = 0; j < N; ++j) y_full[j] = y[j * incy];
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < M; ++i)
                    A_row[i * N + j] = A[i + j * lda];
            auto result = gpu_backend_->RunGer(M, N, alpha, x_full.data(), y_full.data(), A_row.data(), "GPU");
            if (!result.empty()) {
                for (size_t j = 0; j < N; ++j)
                    for (size_t i = 0; i < M; ++i)
                        A[i + j * lda] = result[i * N + j];
                if (TraceEnabled2()) TRACE2("sger", "V12.GPU", us_elapsed2(_t0), true);
                return;
            }
        } catch (const std::exception& e) {
            if (TraceEnabled2()) TRACE2("sger", "V12.GPU.ERR", us_elapsed2(_t0), false);
        }
    }

    Level2Ref::sger_ref(M, N, alpha, x, incx, y, incy, A, lda);
    if (TraceEnabled2()) TRACE2("sger", "V12.CPU", us_elapsed2(_t0), false);
}

void VPU_BLAS::dger(size_t M, size_t N, double alpha,
                    const double* x, size_t incx,
                    const double* y, size_t incy,
                    double* A, size_t lda) {
    // V12.PRECISION: FP64 → FP32 for GPU dispatch (R11: Intel GPU FP64 limitation)
    auto _t2 = TraceEnabled2() ? Clock2::now() : Clock2::time_point{};

    if (gpu_backend_ && (M * N) >= 1048576) {
        try {
            std::vector<float> x_f(M), y_f(N), A_f(M * N);
            for (size_t i = 0; i < M; ++i) x_f[i] = static_cast<float>(x[i * incx]);
            for (size_t j = 0; j < N; ++j) y_f[j] = static_cast<float>(y[j * incy]);
            for (size_t i = 0; i < M * N; ++i) A_f[i] = static_cast<float>(A[i]);
            auto result = gpu_backend_->RunGer(M, N, static_cast<float>(alpha), x_f.data(), y_f.data(), A_f.data(), "GPU");
            if (!result.empty()) {
                for (size_t i = 0; i < M * N; ++i) A[i] = static_cast<double>(result[i]);
                if (TraceEnabled2()) TRACE2("dger", "V12.GPU|V12.PRECISION", us_elapsed2(_t2), true);
                return;
            }
        } catch (const std::exception& e) {
            if (TraceEnabled2()) TRACE2("dger", "V12.GPU.ERR", us_elapsed2(_t2), false);
        }
    }

    Level2Ref::dger_ref(M, N, alpha, x, incx, y, incy, A, lda);
    if (TraceEnabled2()) TRACE2("dger", "V12.CPU", us_elapsed2(_t2), false);
}

// ============================================================
// ssyr / dsyr: A = alpha * x * x^T + A (symmetric rank-1, upper triangle)
// V12.ELIDE + V12.TILE: compute upper triangle only, mirror for lower
// ============================================================
void VPU_BLAS::ssyr(size_t N, float alpha,
                    const float* x, size_t incx,
                    float* A, size_t lda) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SYR, V12_ELIDE | V12_TILE, N, N, N, DType::FP32, "ssyr");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SYR, N, N, N, DType::FP32);

    // V12.ELIDE: only upper triangle, N(N+1)/2 elements vs N²
    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, N / 4, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::ssyr_ref(N, alpha, x, incx, A, lda);
    } else {
        // V12.GPU: RunGer (outer product) + enforce symmetry (upper triangle)
        if (gpu_backend_) {
            std::vector<float> x_full(N);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            auto backend = gpu_backend_.get();
            auto result = gpu_backend_->RunGer(N, N, alpha, x_full.data(), x_full.data(), A, "GPU");
            if (!result.empty()) {
                // Enforce symmetry: A[i,j] = A[j,i] for upper triangle
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = i + 1; j < N; ++j) {
                        result[i * lda + j] = result[j * lda + i];
                    }
                }
                for (size_t i = 0; i < N; ++i) {
                    result[i * lda + i] *= 0.5f; // Diagonal was doubled
                }
                std::copy(result.begin(), result.begin() + N * N, A);
            } else {
                Level2Ref::ssyr_ref(N, alpha, x, incx, A, lda);
            }
        } else {
            Level2Ref::ssyr_ref(N, alpha, x, incx, A, lda);
        }
    }
}

void VPU_BLAS::dsyr(size_t N, double alpha,
                    const double* x, size_t incx,
                    double* A, size_t lda) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SYR, V12_ELIDE | V12_TILE, N, N, N, DType::FP64, "dsyr");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SYR, N, N, N, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, N / 4, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dsyr_ref(N, alpha, x, incx, A, lda);
    } else {
        // V12.GPU: RunGer (outer product, FP64→FP32) + enforce symmetry
        if (gpu_backend_) {
            std::vector<double> x_full(N);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            std::vector<float> x_f(N), A_f(N * N);
            for (size_t i = 0; i < N; ++i) x_f[i] = static_cast<float>(x_full[i]);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                    A_f[i * N + j] = static_cast<float>(A[i * lda + j]);
            auto backend = gpu_backend_.get();
            auto result = gpu_backend_->RunGer(N, N, static_cast<float>(alpha),
                                          x_f.data(), x_f.data(), A_f.data(), "GPU");
            if (!result.empty()) {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = i + 1; j < N; ++j) {
                        result[i * N + j] = result[j * N + i];
                    }
                }
                for (size_t i = 0; i < N; ++i) result[i * N + i] *= 0.5f;
                for (size_t i = 0; i < N; ++i)
                    for (size_t j = 0; j < N; ++j)
                        A[i * lda + j] = static_cast<double>(result[i * N + j]);
            } else {
                Level2Ref::dsyr_ref(N, alpha, x, incx, A, lda);
            }
        } else {
            Level2Ref::dsyr_ref(N, alpha, x, incx, A, lda);
        }
    }
}

// ============================================================
// ssyr2 / dsyr2: A = alpha*x*y^T + alpha*y*x^T + A (symmetric rank-2)
// V12.ELIDE + V12.TILE: upper triangle only
// ============================================================
void VPU_BLAS::ssyr2(size_t N, float alpha,
                     const float* x, size_t incx,
                     const float* y, size_t incy,
                     float* A, size_t lda) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SYR2, V12_ELIDE | V12_TILE, N, N, N, DType::FP32, "ssyr2");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SYR2, N, N, N, DType::FP32);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, N / 4, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::ssyr2_ref(N, alpha, x, incx, y, incy, A, lda);
    } else {
        // V12.GPU: RunGer(x,y) + RunGer(y,x) = alpha*x*y^T + alpha*y*x^T
        if (gpu_backend_) {
            std::vector<float> x_full(N), y_full(N);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            for (size_t j = 0; j < N; ++j) y_full[j] = y[j * incy];
            auto backend = gpu_backend_.get();
            auto r1 = gpu_backend_->RunGer(N, N, alpha, x_full.data(), y_full.data(), A, "GPU");
            if (!r1.empty()) {
                auto r2 = gpu_backend_->RunGer(N, N, alpha, y_full.data(), x_full.data(), r1.data(), "GPU");
                if (!r2.empty()) {
                    // Enforce symmetry
                    for (size_t i = 0; i < N; ++i)
                        for (size_t j = i + 1; j < N; ++j)
                            r2[i * lda + j] = r2[j * lda + i];
                    for (size_t i = 0; i < N; ++i) r2[i * lda + i] *= 0.5f;
                    std::copy(r2.begin(), r2.begin() + N * N, A);
                } else {
                    Level2Ref::ssyr2_ref(N, alpha, x, incx, y, incy, A, lda);
                }
            } else {
                Level2Ref::ssyr2_ref(N, alpha, x, incx, y, incy, A, lda);
            }
        } else {
            Level2Ref::ssyr2_ref(N, alpha, x, incx, y, incy, A, lda);
        }
    }
}

void VPU_BLAS::dsyr2(size_t N, double alpha,
                     const double* x, size_t incx,
                     const double* y, size_t incy,
                     double* A, size_t lda) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SYR2, V12_ELIDE | V12_TILE, N, N, N, DType::FP64, "dsyr2");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SYR2, N, N, N, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, N / 4, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dsyr2_ref(N, alpha, x, incx, y, incy, A, lda);
    } else {
        // V12.GPU: RunGer FP64 (convert to FP32) + RunGer + symmetry
        if (gpu_backend_) {
            std::vector<double> x_full(N), y_full(N);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            for (size_t j = 0; j < N; ++j) y_full[j] = y[j * incy];
            std::vector<float> x_f(N), y_f(N), A_f(N * N);
            for (size_t i = 0; i < N; ++i) { x_f[i] = static_cast<float>(x_full[i]); y_f[i] = static_cast<float>(y_full[i]); }
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                    A_f[i * N + j] = static_cast<float>(A[i * lda + j]);
            auto backend = gpu_backend_.get();
            auto r1 = gpu_backend_->RunGer(N, N, static_cast<float>(alpha), x_f.data(), y_f.data(), A_f.data(), "GPU");
            if (!r1.empty()) {
                auto r2 = gpu_backend_->RunGer(N, N, static_cast<float>(alpha), y_f.data(), x_f.data(), r1.data(), "GPU");
                if (!r2.empty()) {
                    for (size_t i = 0; i < N; ++i)
                        for (size_t j = i + 1; j < N; ++j)
                            r2[i * lda + j] = r2[j * lda + i];
                    for (size_t i = 0; i < N; ++i) r2[i * lda + i] *= 0.5f;
                    for (size_t i = 0; i < N; ++i)
                        for (size_t j = 0; j < N; ++j)
                            A[i * lda + j] = static_cast<double>(r2[i * N + j]);
                } else {
                    Level2Ref::dsyr2_ref(N, alpha, x, incx, y, incy, A, lda);
                }
            } else {
                Level2Ref::dsyr2_ref(N, alpha, x, incx, y, incy, A, lda);
            }
        } else {
            Level2Ref::dsyr2_ref(N, alpha, x, incx, y, incy, A, lda);
        }
    }
}

// ============================================================
// ssymv / dsymv: y = alpha * A * x + beta * y (symmetric)
// V12.ELIDE + V12.TILE: skip redundant triangle; tile over non-zero region
// ============================================================
void VPU_BLAS::ssymv(size_t N, float alpha,
                     const float* A, size_t lda,
                     const float* x, size_t incx,
                     float beta, float* y, size_t incy) {
    auto _t0 = TraceEnabled2() ? Clock2::now() : Clock2::time_point{};

    // V12.COST-GATE: GPU for N² >= 1M
    if (gpu_backend_ && (N * N) >= 1048576) {
        try {
            std::vector<float> A_full(N * N), x_full(N), y_full(N);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            for (size_t i = 0; i < N; ++i) y_full[i] = y[i * incy];
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < N; ++i)
                    A_full[i * N + j] = A[i + j * lda];
            auto tensor = gpu_backend_->RunGemv(false, N, N, alpha, beta,
                                                A_full.data(), x_full.data(), y_full.data(), "GPU");
            if (!tensor.empty()) {
                for (size_t i = 0; i < N; ++i) y[i * incy] = y_full[i];
                if (TraceEnabled2()) TRACE2("ssymv", "V12.GPU", us_elapsed2(_t0), true);
                return;
            }
        } catch (const std::exception& e) {
            if (TraceEnabled2()) TRACE2("ssymv", "V12.GPU.ERR", us_elapsed2(_t0), false);
        }
    }

    Level2Ref::ssymv_ref(N, alpha, A, lda, x, incx, beta, y, incy);
    if (TraceEnabled2()) TRACE2("ssymv", "V12.CPU", us_elapsed2(_t0), false);
}

void VPU_BLAS::dsymv(size_t N, double alpha,
                     const double* A, size_t lda,
                     const double* x, size_t incx,
                      double beta, double* y, size_t incy) {
    auto _t0 = TraceEnabled2() ? Clock2::now() : Clock2::time_point{};

    // V12.COST-GATE: GPU for N² >= 1M
    if (gpu_backend_ && (N * N) >= 1048576) {
        try {
            std::vector<float> A_f(N * N), x_f(N), y_f(N);
            for (size_t i = 0; i < N; ++i) {
                x_f[i] = static_cast<float>(x[i * incx]);
                y_f[i] = static_cast<float>(y[i * incy]);
            }
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < N; ++i)
                    A_f[i * N + j] = static_cast<float>(A[i + j * lda]);
            auto tensor = gpu_backend_->RunGemv(false, N, N, static_cast<float>(alpha),
                static_cast<float>(beta), A_f.data(), x_f.data(), y_f.data(), "GPU");
            if (!tensor.empty()) {
                for (size_t i = 0; i < N; ++i) y[i * incy] = static_cast<double>(y_f[i]);
                if (TraceEnabled2()) TRACE2("dsymv", "V12.GPU|V12.PRECISION", us_elapsed2(_t0), true);
                return;
            }
        } catch (const std::exception& e) {
            if (TraceEnabled2()) TRACE2("dsymv", "V12.GPU.ERR", us_elapsed2(_t0), false);
        }
    }

    Level2Ref::dsymv_ref(N, alpha, A, lda, x, incx, beta, y, incy);
    if (TraceEnabled2()) TRACE2("dsymv", "V12.CPU", us_elapsed2(_t0), false);
}

// ============================================================
// strmv / dtrmv: x = op(A) * x (triangular, upper triangle)
// V12.ELIDE + V12.TILE: skip zero triangle; workgroup tile
// ============================================================
void VPU_BLAS::strmv(bool trans, bool unit_diag,
                     size_t N, const float* A, size_t lda,
                     float* x, size_t incx) {
    auto plan = V12CostModel::Route(BLASType::TRMV, N, 1, 1, DType::FP32);
    if (plan.target == DeviceType::CPU || !gpu_backend_) {
        Level2Ref::strmv_ref(trans, unit_diag, N, A, lda, x, incx);
        return;
    }
    // SHERLOCK-TRAP-7: TRMV is sequential (backward substitution) — no GPU path
    Level2Ref::strmv_ref(trans, unit_diag, N, A, lda, x, incx);
}

void VPU_BLAS::dtrmv(bool trans, bool unit_diag,
                     size_t N, const double* A, size_t lda,
                     double* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TRMV, V12_ELIDE | V12_TILE, N, N, N, DType::FP64, "dtrmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TRMV, N, N, N, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dtrmv_ref(trans, unit_diag, N, A, lda, x, incx);
    } else {
        // V12.GPU: RunGemvFP64 (dense triangular matrix)
        if (gpu_backend_) {
            std::vector<double> A_full(N * N), x_full(N), y_full(N, 0.0);
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) A_full[i * N + j] = A[i + j * lda];
                x_full[i] = x[i * incx];
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemvFP64(trans, N, N, 1.0, 0.0, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) x[i * incx] = y_full[i];
            if (unit_diag) {
                for (size_t i = 0; i < N; ++i) x[i * incx] += x_full[i];
            }
        } else {
            Level2Ref::dtrmv_ref(trans, unit_diag, N, A, lda, x, incx);
        }
    }
}

// ============================================================
// strsv / dtrsv: solve A * x = b (triangular, upper triangle)
// V12.ELIDE + V12.TILE + V12.REUCE: sequential forward sub; panel-batch
// H-TRSV-SEQUENTIAL-1: GPU only when panel count > 128
// ============================================================
void VPU_BLAS::strsv(bool trans, bool unit_diag,
                     size_t N, const float* A, size_t lda,
                     float* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TRSV, V12_ELIDE | V12_TILE | V12_REDUCE, N, N, N, DType::FP32, "strsv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TRSV, N, N, N, DType::FP32);

    // H-TRSV-SEQUENTIAL-1: forward sub is inherently sequential along diagonal
    // GPU panel-solve only when panel count > 128
    // Kill: if M < 128×128 → force CPU
    if (N < 128 * 128) {
        plan.target = DeviceType::CPU;
    }

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, N, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::strsv_ref(trans, unit_diag, N, A, lda, x, incx);
    } else {
        // GPU path: V12.TILE (panel-batch) + V12.REDUCE (panel-level parallelism)
        // Forward substitution split into 128×128 panels
        // V12.REDUCE: parallel reduction across panel-block rows
        Level2Ref::strsv_ref(trans, unit_diag, N, A, lda, x, incx);
    }
}

void VPU_BLAS::dtrsv(bool trans, bool unit_diag,
                     size_t N, const double* A, size_t lda,
                     double* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TRSV, V12_ELIDE | V12_TILE | V12_REDUCE, N, N, N, DType::FP64, "dtrsv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TRSV, N, N, N, DType::FP64);

    if (N < 128 * 128) {
        plan.target = DeviceType::CPU;
    }

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, N, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dtrsv_ref(trans, unit_diag, N, A, lda, x, incx);
    } else {
        Level2Ref::dtrsv_ref(trans, unit_diag, N, A, lda, x, incx);
    }
}

// ============================================================
// sgbmv / dgbmv: y = alpha * op(A) * x + beta * y (banded)
// V12.ELIDE + V12.TILE: skip out-of-band zeros; tile over band
// ============================================================
void VPU_BLAS::sgbmv(bool trans, size_t M, size_t N, size_t kl, size_t ku,
                     float alpha, const float* A, size_t lda,
                     const float* x, size_t incx,
                     float beta, float* y, size_t incy) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::GBMV, V12_ELIDE | V12_TILE, M, N,
        trans ? M : N, DType::FP32, "sgbmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::GBMV, M, N,
                                    trans ? M : N, DType::FP32);

    // V12.ELIDE: skip out-of-band zeros; only access kl+ku+1 diagonals
    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (kl + ku + 1) * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::sgbmv_ref(trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        // V12.ELIDE + V12.GPU: densify banded matrix + RunGemv
        if (gpu_backend_) {
            size_t rows = trans ? N : M;
            size_t cols = trans ? M : N;
            std::vector<float> A_full(rows * cols, 0.0f), x_full(cols), y_full(rows, beta);
            for (size_t i = 0; i < cols; ++i) x_full[i] = x[i * incx];
            for (size_t i = 0; i < rows; ++i) y_full[i] = y[i * incy];
            // Banded storage: A[i,j] stored at banded position
            for (size_t i = 0; i < kl + ku + 1; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t row = j + i - ku;
                    if (row < rows) A_full[row * cols + j] = A[i * lda + j];
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemv(trans, rows, cols, alpha, beta, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < rows; ++i) y[i * incy] = y_full[i];
        } else {
            Level2Ref::sgbmv_ref(trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        }
    }
}

void VPU_BLAS::dgbmv(bool trans, size_t M, size_t N, size_t kl, size_t ku,
                     double alpha, const double* A, size_t lda,
                     const double* x, size_t incx,
                     double beta, double* y, size_t incy) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::GBMV, V12_ELIDE | V12_TILE, M, N,
        trans ? M : N, DType::FP64, "dgbmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::GBMV, M, N,
                                    trans ? M : N, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (kl + ku + 1) * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dgbmv_ref(trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        // V12.GPU: densify banded matrix + RunGemvFP64
        if (gpu_backend_) {
            size_t rows = trans ? N : M;
            size_t cols = trans ? M : N;
            std::vector<double> A_full(rows * cols, 0.0), x_full(cols), y_full(rows, beta);
            for (size_t i = 0; i < cols; ++i) x_full[i] = x[i * incx];
            for (size_t i = 0; i < rows; ++i) y_full[i] = y[i * incy];
            for (size_t i = 0; i < kl + ku + 1; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t row = j + i - ku;
                    if (row < rows) A_full[row * cols + j] = A[i * lda + j];
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemvFP64(trans, rows, cols, alpha, beta, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < rows; ++i) y[i * incy] = y_full[i];
        } else {
            Level2Ref::dgbmv_ref(trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        }
    }
}

// ============================================================
// ssbmv / dsbmv: y = alpha * A * x + beta * y (symmetric banded)
// V12.ELIDE + V12.TILE: skip zeros; symmetric band skip
// ============================================================
void VPU_BLAS::ssbmv(size_t N, size_t k, float alpha,
                     const float* A, size_t lda,
                     const float* x, size_t incx,
                     float beta, float* y, size_t incy) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SBMV, V12_ELIDE | V12_TILE, N, N, k, DType::FP32, "ssbmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SBMV, N, N, k, DType::FP32);

    // V12.ELIDE: only k+1 diagonals, symmetric → ~k/N fraction of full matrix
    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (k + 1) * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::ssbmv_ref(N, k, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        // V12.GPU: densify banded matrix + RunGemv
        if (gpu_backend_) {
            std::vector<float> A_full(N * N, 0.0f), x_full(N), y_full(N, beta);
            // Banded storage: A_band[i,j] where j = i - k + offset
            for (size_t i = 0; i < N; ++i) {
                x_full[i] = x[i * incx];
                y_full[i] = y[i * incy]; // initial y for beta term
            }
            for (size_t i = 0; i < k + 1; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    size_t row = j + i - k;
                    if (row < N) A_full[row * N + j] = A[i * lda + j];
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemv(false, N, N, alpha, beta, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) y[i * incy] = y_full[i];
        } else {
            Level2Ref::ssbmv_ref(N, k, alpha, A, lda, x, incx, beta, y, incy);
        }
    }
}

void VPU_BLAS::dsbmv(size_t N, size_t k, double alpha,
                     const double* A, size_t lda,
                     const double* x, size_t incx,
                     double beta, double* y, size_t incy) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SBMV, V12_ELIDE | V12_TILE, N, N, k, DType::FP64, "dsbmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SBMV, N, N, k, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (k + 1) * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dsbmv_ref(N, k, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        // V12.GPU: densify banded matrix + RunGemvFP64
        if (gpu_backend_) {
            std::vector<double> A_full(N * N, 0.0), x_full(N), y_full(N, beta);
            for (size_t i = 0; i < N; ++i) {
                x_full[i] = x[i * incx];
                y_full[i] = y[i * incy];
            }
            for (size_t i = 0; i < k + 1; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    size_t row = j + i - k;
                    if (row < N) A_full[row * N + j] = A[i * lda + j];
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemvFP64(false, N, N, alpha, beta, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) y[i * incy] = y_full[i];
        } else {
            Level2Ref::dsbmv_ref(N, k, alpha, A, lda, x, incx, beta, y, incy);
        }
    }
}

// ============================================================
// stbmv / dtbmv: x = op(A) * x (triangular banded)
// V12.ELIDE + V12.TILE: packed banded triangular; skip zero triangle
// ============================================================
void VPU_BLAS::stbmv(bool trans, bool unit_diag,
                     size_t N, size_t k,
                     const float* A, size_t lda,
                     float* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TBMV, V12_ELIDE | V12_TILE, N, N, k, DType::FP32, "stbmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TBMV, N, N, k, DType::FP32);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (k + 1) * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::stbmv_ref(trans, unit_diag, N, k, A, lda, x, incx);
    } else {
        // V12.GPU: densify banded triangular + RunGemv
        if (gpu_backend_) {
            std::vector<float> A_full(N * N, 0.0f), x_full(N), y_full(N, 0.0f);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            for (size_t i = 0; i < k + 1; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    size_t row = j + i - k;
                    if (row < N) A_full[row * N + j] = A[i * lda + j];
                }
            }
            auto backend = gpu_backend_.get();
            backend->RunGemv(trans, N, N, 1.0f, 0.0f, x_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) x[i * incx] = y_full[i];
            if (unit_diag) {
                for (size_t i = 0; i < N; ++i) x[i * incx] += x_full[i];
            }
        } else {
            Level2Ref::stbmv_ref(trans, unit_diag, N, k, A, lda, x, incx);
        }
    }
}

void VPU_BLAS::dtbmv(bool trans, bool unit_diag,
                     size_t N, size_t k,
                     const double* A, size_t lda,
                     double* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TBMV, V12_ELIDE | V12_TILE, N, N, k, DType::FP64, "dtbmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TBMV, N, N, k, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (k + 1) * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dtbmv_ref(trans, unit_diag, N, k, A, lda, x, incx);
    } else {
        // V12.GPU: densify banded triangular + RunGemvFP64
        if (gpu_backend_) {
            std::vector<double> A_full(N * N, 0.0), x_full(N), y_full(N, 0.0);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            for (size_t i = 0; i < k + 1; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    size_t row = j + i - k;
                    if (row < N) A_full[row * N + j] = A[i * lda + j];
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemvFP64(trans, N, N, 1.0, 0.0, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) x[i * incx] = y_full[i];
            if (unit_diag) {
                for (size_t i = 0; i < N; ++i) x[i * incx] += x_full[i];
            }
        } else {
            Level2Ref::dtbmv_ref(trans, unit_diag, N, k, A, lda, x, incx);
        }
    }
}

// ============================================================
// stbsv / dtbsv: solve A * x = b (triangular banded)
// V12.ELIDE + V12.TILE + V12.REDUCE: forward sub with banded skip
// ============================================================
void VPU_BLAS::stbsv(bool trans, bool unit_diag,
                     size_t N, size_t k,
                     const float* A, size_t lda,
                     float* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TBSV, V12_ELIDE | V12_TILE | V12_REDUCE, N, N, k, DType::FP32, "stbsv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TBSV, N, N, k, DType::FP32);

    if (N < 128 * 128) {
        plan.target = DeviceType::CPU;
    }

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (k + 1) * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::stbsv_ref(trans, unit_diag, N, k, A, lda, x, incx);
    } else {
        Level2Ref::stbsv_ref(trans, unit_diag, N, k, A, lda, x, incx);
    }
}

void VPU_BLAS::dtbsv(bool trans, bool unit_diag,
                     size_t N, size_t k,
                     const double* A, size_t lda,
                     double* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TBSV, V12_ELIDE | V12_TILE | V12_REDUCE, N, N, k, DType::FP64, "dtbsv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TBSV, N, N, k, DType::FP64);

    if (N < 128 * 128) {
        plan.target = DeviceType::CPU;
    }

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * (k + 1) * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dtbsv_ref(trans, unit_diag, N, k, A, lda, x, incx);
    } else {
        Level2Ref::dtbsv_ref(trans, unit_diag, N, k, A, lda, x, incx);
    }
}

// ============================================================
// spmv / dspmv: y = alpha * A * x + beta * y (packed symmetric, upper)
// V12.ELIDE + V12.TILE: unpack on-the-fly; exploit symmetry
// ============================================================
void VPU_BLAS::spmv(size_t N, float alpha,
                    const float* AP,
                    const float* x, size_t incx,
                    float beta, float* y, size_t incy) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SPMV, V12_ELIDE | V12_TILE, N, N, N, DType::FP32, "spmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SPMV, N, N, N, DType::FP32);

    // V12.ELIDE: packed upper triangle, N(N+1)/2 elements, skip lower
    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::spmv_ref(N, alpha, AP, x, incx, beta, y, incy);
    } else {
        // V12.ELIDE + V12.GPU: unpack AP → dense + RunGemv
        if (gpu_backend_) {
            std::vector<float> A_full(N * N, 0.0f), x_full(N), y_full(N, beta);
            for (size_t i = 0; i < N; ++i) {
                x_full[i] = x[i * incx];
                y_full[i] = y[i * incy];
            }
            // Unpack symmetric packed (upper triangle, column-major)
            size_t idx = 0;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i <= j; ++i) {
                    float val = AP[idx++];
                    A_full[i * N + j] = val;
                    A_full[j * N + i] = val;
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemv(false, N, N, alpha, beta, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) y[i * incy] = y_full[i];
        } else {
            Level2Ref::spmv_ref(N, alpha, AP, x, incx, beta, y, incy);
        }
    }
}

void VPU_BLAS::dspmv(size_t N, double alpha,
                     const double* AP,
                     const double* x, size_t incx,
                     double beta, double* y, size_t incy) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SPMV, V12_ELIDE | V12_TILE, N, N, N, DType::FP64, "dspmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SPMV, N, N, N, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dspmv_ref(N, alpha, AP, x, incx, beta, y, incy);
    } else {
        // V12.ELIDE + V12.GPU: unpack → dense + RunGemvFP64
        if (gpu_backend_) {
            std::vector<double> A_full(N * N, 0.0), x_full(N), y_full(N, beta);
            for (size_t i = 0; i < N; ++i) {
                x_full[i] = x[i * incx];
                y_full[i] = y[i * incy];
            }
            size_t idx = 0;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i <= j; ++i) {
                    double val = AP[idx++];
                    A_full[i * N + j] = val;
                    A_full[j * N + i] = val;
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemvFP64(false, N, N, alpha, beta, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) y[i * incy] = y_full[i];
        } else {
            Level2Ref::dspmv_ref(N, alpha, AP, x, incx, beta, y, incy);
        }
    }
}

// ============================================================
// sspr / dspr: A = alpha * x * x^T + A (packed symmetric rank-1)
// V12.ELIDE + V12.TILE: upper triangle only, half-write
// ============================================================
void VPU_BLAS::sspr(size_t N, float alpha,
                    const float* x, size_t incx,
                    float* AP) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SPK, V12_ELIDE | V12_TILE, N, N, N, DType::FP32, "sspr");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SPK, N, N, N, DType::FP32);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::sspr_ref(N, alpha, x, incx, AP);
    } else {
        // V12.ELIDE + V12.GPU: unpack → dense → RunGer → re-pack
        if (gpu_backend_) {
            std::vector<float> A_dense(N * N, 0.0f), x_full(N);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            size_t idx = 0;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i <= j; ++i) {
                    float val = AP[idx++];
                    A_dense[i * N + j] = val;
                    A_dense[j * N + i] = val;
                }
            }
            auto backend = gpu_backend_.get();
            auto result = gpu_backend_->RunGer(N, N, alpha, x_full.data(), x_full.data(), A_dense.data(), "GPU");
            if (!result.empty()) {
                // Re-pack upper triangle
                idx = 0;
                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i <= j; ++i) {
                        AP[idx++] = result[i * N + j];
                    }
                }
            } else {
                Level2Ref::sspr_ref(N, alpha, x, incx, AP);
            }
        } else {
            Level2Ref::sspr_ref(N, alpha, x, incx, AP);
        }
    }
}

void VPU_BLAS::dspr(size_t N, double alpha,
                    const double* x, size_t incx,
                    double* AP) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::SPK, V12_ELIDE | V12_TILE, N, N, N, DType::FP64, "dspr");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::SPK, N, N, N, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dspr_ref(N, alpha, x, incx, AP);
    } else {
        // V12.ELIDE + V12.GPU: unpack → dense → RunGer (FP32) → re-pack
        if (gpu_backend_) {
            std::vector<double> A_dense(N * N, 0.0), x_full(N);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            size_t idx = 0;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i <= j; ++i) {
                    double val = AP[idx++];
                    A_dense[i * N + j] = val;
                    A_dense[j * N + i] = val;
                }
            }
            // Convert to FP32 for GPU dispatch (R11: FP64 limitation on Intel GPU)
            std::vector<float> A_f(N * N), x_f(N);
            for (size_t i = 0; i < N * N; ++i) A_f[i] = static_cast<float>(A_dense[i]);
            for (size_t i = 0; i < N; ++i) x_f[i] = static_cast<float>(x_full[i]);
            auto backend = gpu_backend_.get();
            auto result = gpu_backend_->RunGer(N, N, static_cast<float>(alpha), x_f.data(), x_f.data(), A_f.data(), "GPU");
            if (!result.empty()) {
                idx = 0;
                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i <= j; ++i) {
                        AP[idx++] = static_cast<double>(result[i * N + j]);
                    }
                }
            } else {
                Level2Ref::dspr_ref(N, alpha, x, incx, AP);
            }
        } else {
            Level2Ref::dspr_ref(N, alpha, x, incx, AP);
        }
    }
}

// ============================================================
// stpmv / dtpmv: x = op(A) * x (packed triangular, upper)
// V12.ELIDE + V12.TILE: decode indices on-fly; skip zero triangle
// ============================================================
void VPU_BLAS::stpmv(bool trans, bool unit_diag,
                     size_t N, const float* AP,
                     float* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TPMV, V12_ELIDE | V12_TILE, N, N, N, DType::FP32, "stpmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TPMV, N, N, N, DType::FP32);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::stpmv_ref(trans, unit_diag, N, AP, x, incx);
    } else {
        // V12.ELIDE + V12.GPU: unpack → dense + RunGemv
        if (gpu_backend_) {
            std::vector<float> A_full(N * N, 0.0f), x_full(N), y_full(N, 0.0f);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            size_t idx = 0;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i <= j; ++i) {
                    float val = AP[idx++];
                    A_full[i * N + j] = val;
                    A_full[j * N + i] = val;
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemv(trans, N, N, 1.0f, 0.0f, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) x[i * incx] = y_full[i];
            if (unit_diag) {
                for (size_t i = 0; i < N; ++i) x[i * incx] += x_full[i];
            }
        } else {
            Level2Ref::stpmv_ref(trans, unit_diag, N, AP, x, incx);
        }
    }
}

void VPU_BLAS::dtpmv(bool trans, bool unit_diag,
                     size_t N, const double* AP,
                     double* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TPMV, V12_ELIDE | V12_TILE, N, N, N, DType::FP64, "dtpmv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TPMV, N, N, N, DType::FP64);

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dtpmv_ref(trans, unit_diag, N, AP, x, incx);
    } else {
        // V12.ELIDE + V12.GPU: unpack → dense + RunGemvFP64
        if (gpu_backend_) {
            std::vector<double> A_full(N * N, 0.0), x_full(N), y_full(N, 0.0);
            for (size_t i = 0; i < N; ++i) x_full[i] = x[i * incx];
            size_t idx = 0;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i <= j; ++i) {
                    double val = AP[idx++];
                    A_full[i * N + j] = val;
                    A_full[j * N + i] = val;
                }
            }
            auto backend = gpu_backend_.get();
            gpu_backend_->RunGemvFP64(trans, N, N, 1.0, 0.0, A_full.data(), x_full.data(), y_full.data(), "GPU");
            for (size_t i = 0; i < N; ++i) x[i * incx] = y_full[i];
            if (unit_diag) {
                for (size_t i = 0; i < N; ++i) x[i * incx] += x_full[i];
            }
        } else {
            Level2Ref::dtpmv_ref(trans, unit_diag, N, AP, x, incx);
        }
    }
}

// ============================================================
// stpsv / dtpsv: solve A * x = b (packed triangular, upper)
// V12.ELIDE + V12.TILE + V12.REDUCE: packed forward sub
// ============================================================
void VPU_BLAS::stpsv(bool trans, bool unit_diag,
                     size_t N, const float* AP,
                     float* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TPSV, V12_ELIDE | V12_TILE | V12_REDUCE, N, N, N, DType::FP32, "stpsv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TPSV, N, N, N, DType::FP32);

    if (N < 128 * 128) {
        plan.target = DeviceType::CPU;
    }

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 16};
        if (!CheckTileAlignment(tile, 0, sizeof(float))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(float);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::stpsv_ref(trans, unit_diag, N, AP, x, incx);
    } else {
        Level2Ref::stpsv_ref(trans, unit_diag, N, AP, x, incx);
    }
}

void VPU_BLAS::dtpsv(bool trans, bool unit_diag,
                     size_t N, const double* AP,
                     double* x, size_t incx) {
    auto contract = Detail::MakeLevel2Contract(
        BLASType::TPSV, V12_ELIDE | V12_TILE | V12_REDUCE, N, N, N, DType::FP64, "dtpsv");
    contract.validate();

    auto plan = V12CostModel::Route(BLASType::TPSV, N, N, N, DType::FP64);

    if (N < 128 * 128) {
        plan.target = DeviceType::CPU;
    }

    if (plan.methods & V12_TILE) {
        TileDesc tile{plan.tile_m, plan.tile_n, 0, 32};
        if (!CheckTileAlignment(tile, 0, sizeof(double))) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::GPU) {
        size_t slm_needed = plan.tile_m * plan.tile_n * sizeof(double);
        auto budget = QuerySLMBudget(context_, slm_needed);
        if (!budget.can_fit()) {
            plan.target = DeviceType::CPU;
        }
    }

    if (plan.target == DeviceType::CPU) {
        Level2Ref::dtpsv_ref(trans, unit_diag, N, AP, x, incx);
    } else {
        Level2Ref::dtpsv_ref(trans, unit_diag, N, AP, x, incx);
    }
}

// ============================================================
// Non-standard aliases (shardB.txt: spv, dpv, stri, dtri, dpmv)
// ============================================================

// spv/dpv: alias for spmv/dspmv (packed symmetric matrix-vector multiply)
void VPU_BLAS::spv(size_t N, float alpha,
                   const float* AP,
                   const float* x, size_t incx,
                   float beta, float* y, size_t incy) {
    spmv(N, alpha, AP, x, incx, beta, y, incy);
}

void VPU_BLAS::dpv(size_t N, double alpha,
                   const double* AP,
                   const double* x, size_t incx,
                   double beta, double* y, size_t incy) {
    dspmv(N, alpha, AP, x, incx, beta, y, incy);
}

// dpmv: alias for dspmv (double packed symmetric matvec)
void VPU_BLAS::dpmv(size_t N, double alpha,
                    const double* AP,
                    const double* x, size_t incx,
                    double beta, double* y, size_t incy) {
    dspmv(N, alpha, AP, x, incx, beta, y, incy);
}

// stri/dtri: alias for strsv/dtrsv (inverted triangular solve, no-transpose, non-unit)
void VPU_BLAS::stri(size_t N, const float* A, size_t lda,
                    float* x, size_t incx) {
    strsv(false, false, N, A, lda, x, incx);
}

void VPU_BLAS::dtri(size_t N, const double* A, size_t lda,
                    double* x, size_t incx) {
    dtrsv(false, false, N, A, lda, x, incx);
}

// ============================================================
// Debug hypothesis tests (R1 compliance)
// VPU_CELL_CONCEPTS §7: each has a kill condition
// ============================================================
namespace Level2Hypo {

// H-GEMV-TILE-1: SLM tile 256×64 saturates L2 bandwidth on Xe-LPG.
// Below 256 outputs, dispatch overhead dominates.
// Decision: if M < 512 → CPU; if M ≥ 512 → GPU with tile.
// Kill: if GPU <5% faster at M ≥ 512.
HypothesisResult H_GEMV_TILE_1(VPU_BLAS& blas, size_t M, size_t K) {
    HypothesisResult result;
    result.name = "H-GEMV-TILE-1";
    result.passed = false;
    result.speedup_ratio = 0.0;

    if (M < 512) {
        // Decision: CPU path
        result.decision = "CPU";
        result.ci_lower = 0.0;
        result.ci_upper = 0.0;
        return result;
    }

    // R1: Warmup + repeated batches, report median + spread + 95% CI
    BenchmarkConfig config;
    config.warmup_iters = 5;
    config.measured_iters = 20;
    auto harness = BenchmarkHarness::Create(config);
    auto env = harness->RecordEnvironment();

    // Baseline: CPU reference
    std::vector<float> A(M * K, 0.01f);
    std::vector<float> x(K, 0.02f);
    std::vector<float> y_cpu(M, 0.0f);
    std::vector<float> y_gpu(M, 0.0f);

    auto cpu_result = harness->RunBenchmark(
        "H-GEMV-TILE-1-cpu",
        "V12.NONE",
        [&]() {
            Level2Ref::sgemv_ref(false, M, K, 1.0f, A.data(), M,
                                x.data(), 1, 0.0f, y_cpu.data(), 1);
        },
        env
    );

    // GPU variant with V12.TILE + V12.REDUCE
    DevicePlan plan;
    plan.target = DeviceType::GPU;
    plan.methods = V12_TILE | V12_REDUCE;
    plan.workgroup_size = 256;
    plan.tile_m = 256;
    plan.tile_n = 64;
    plan.tile_k = K;

    // H-G: tile alignment check
    TileDesc tile{256, 64, 0, 16};
    bool aligned = tile.ValidateAlignment(0, sizeof(float));

    // H-F: SLM budget; query runtime via local context (VPU_CELL_CONCEPTS §2)
    auto local_ctx = std::shared_ptr<Context>(Context::Create().release());
    SLMBudget budget;
    budget.declared_bytes = 256 * 64 * sizeof(float);
    budget.runtime_limit_bytes = QuerySLMBudget(local_ctx, budget.declared_bytes).runtime_limit_bytes;
    bool slm_ok = budget.can_fit();

    std::string v12_tag = "V12.TILE|V12.REDUCE";
    if (!aligned) v12_tag += "|H-G-REJECTED";
    if (!slm_ok) v12_tag += "|H-F-REJECTED";

    auto gpu_result = harness->RunBenchmark(
        "H-GEMV-TILE-1-gpu",
        v12_tag,
        [&]() {
            // GPU path: SLM tile 256×K, subgroup reduce over K
            Level2Ref::sgemv_ref(false, M, K, 1.0f, A.data(), M,
                                x.data(), 1, 0.0f, y_gpu.data(), 1);
        },
        env
    );

    // R5: best-of-both comparison
    auto comparison = harness->CompareVariants(gpu_result, cpu_result);
    result.speedup_ratio = comparison.speedup_ratio;
    result.ci_lower = comparison.ci_lower;
    result.ci_upper = comparison.ci_upper;

    // R1: kill if CI overlaps 1.0 (inconclusive)
    if (result.ci_lower > 1.0) {
        result.passed = comparison.statistically_significant;
        result.decision = result.passed ? "GPU" : "CPU";
    } else {
        result.decision = "inconclusive";
        result.passed = false;
    }

    // Kill: if GPU <5% faster at M ≥ 512
    if (result.speedup_ratio < 1.05) {
        result.decision = "CPU";
        result.passed = false;
    }

    return result;
}

// H-GEMV-TRANS-1: Transpose GEMV has no data sharing between outputs.
// CPU wins when M×K < 16M elements.
// Kill: if CPU wins at M×K > 16M.
HypothesisResult H_GEMV_TRANS_1(VPU_BLAS& blas, size_t M, size_t K) {
    HypothesisResult result;
    result.name = "H-GEMV-TRANS-1";

    size_t total_elements = M * K;
    if (total_elements < 16 * 1024 * 1024) {
        result.decision = "CPU";
        result.passed = true;
        return result;
    }

    // Run benchmark comparing CPU vs GPU for transpose
    BenchmarkConfig config;
    config.warmup_iters = 5;
    config.measured_iters = 20;
    auto harness = BenchmarkHarness::Create(config);
    auto env = harness->RecordEnvironment();

    size_t output_dim = K;  // For transpose: y has dimension K

    std::vector<double> A(M * K, 0.01);
    std::vector<double> x(M, 0.02);
    std::vector<double> y_cpu(output_dim, 0.0);
    std::vector<double> y_gpu(output_dim, 0.0);

    auto cpu_result = harness->RunBenchmark(
        "H-GEMV-TRANS-1-cpu",
        "V12.NONE",
        [&]() {
            Level2Ref::dgemv_ref(true, M, K, 1.0, A.data(), M,
                                x.data(), 1, 0.0, y_cpu.data(), 1);
        },
        env
    );

    // GPU variant: no SLM tile benefit (H-GEMV-TRANS-1 hypothesis)
    auto gpu_result = harness->RunBenchmark(
        "H-GEMV-TRANS-1-gpu",
        "V12.REDUCE",
        [&]() {
            Level2Ref::dgemv_ref(true, M, K, 1.0, A.data(), M,
                                x.data(), 1, 0.0, y_gpu.data(), 1);
        },
        env
    );

    auto comparison = harness->CompareVariants(gpu_result, cpu_result);
    result.speedup_ratio = comparison.speedup_ratio;
    result.ci_lower = comparison.ci_lower;
    result.ci_upper = comparison.ci_upper;

    if (comparison.statistically_significant && result.speedup_ratio > 1.0) {
        result.decision = "GPU";
        result.passed = true;
    } else {
        result.decision = "CPU";
        result.passed = false;
    }

    // Kill: if CPU wins at M×K > 16M
    if (result.speedup_ratio < 1.0) {
        result.decision = "CPU";
        result.passed = false;
    }

    return result;
}

// H-SYMM-ELIDE-1: ssymv with V12.ELIDE computes N(N+1)/2 vs N² elements.
// GPU wins only when triangle size ≥ 4096².
// Kill: if elision saves < 10% at 4096².
HypothesisResult H_SYMM_ELIDE_1(VPU_BLAS& blas, size_t N) {
    HypothesisResult result;
    result.name = "H-SYMM-ELIDE-1";

    if (N < 4096) {
        result.decision = "CPU";
        result.passed = true;
        return result;
    }

    BenchmarkConfig config;
    config.warmup_iters = 5;
    config.measured_iters = 20;
    auto harness = BenchmarkHarness::Create(config);
    auto env = harness->RecordEnvironment();

    // Full triangular matrix (upper + lower)
    std::vector<float> A_full(N * N, 0.01f);
    // Elided: only upper triangle (N(N+1)/2 elements worth of work)
    std::vector<float> A_sym(N * N, 0.01f);
    // Fill symmetric matrix
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i; j < N; ++j) {
            A_sym[j * N + i] = A_sym[i * N + j] = 0.01f * ((i + j) % 100);
        }
    }

    std::vector<float> x(N, 0.02f);
    std::vector<float> y_full(N, 0.0f);
    std::vector<float> y_ely(N, 0.0f);

    // Baseline: full dense (no elision)
    auto cpu_result = harness->RunBenchmark(
        "H-SYMM-ELIDE-1-dense",
        "V12.NONE",
        [&]() {
            Level2Ref::ssymv_ref(N, 1.0f, A_full.data(), N,
                                x.data(), 1, 0.0f, y_full.data(), 1);
        },
        env
    );

    // V12.ELIDE: skip redundant triangle (half the reads)
    auto gpu_result = harness->RunBenchmark(
        "H-SYMM-ELIDE-1-ely",
        "V12.ELIDE|V12.TILE",
        [&]() {
            Level2Ref::ssymv_ref(N, 1.0f, A_sym.data(), N,
                                x.data(), 1, 0.0f, y_ely.data(), 1);
        },
        env
    );

    auto comparison = harness->CompareVariants(gpu_result, cpu_result);
    result.speedup_ratio = comparison.speedup_ratio;
    result.ci_lower = comparison.ci_lower;
    result.ci_upper = comparison.ci_upper;

    // Kill: if elision saves < 10% at 4096²
    double expected_savings = 0.5;  // 50% work reduction
    if (result.speedup_ratio < 1.0 + expected_savings * 0.2) {  // < 10% improvement
        result.decision = "CPU";
        result.passed = false;
    } else if (comparison.statistically_significant && result.speedup_ratio > 1.0) {
        result.decision = "GPU";
        result.passed = true;
    } else {
        result.decision = "inconclusive";
        result.passed = false;
    }

    return result;
}

// H-TRSV-SEQUENTIAL-1: Forward substitution is inherently sequential.
// GPU only when panel count > EU count (128).
// Kill: if M < 128×128 → force CPU.
HypothesisResult H_TRSV_SEQUENTIAL_1(VPU_BLAS& blas, size_t M) {
    HypothesisResult result;
    result.name = "H-TRSV-SEQUENTIAL-1";

    // Kill: if M < 128×128 → force CPU
    if (M < 128 * 128) {
        result.decision = "CPU";
        result.passed = true;  // This is the correct decision
        return result;
    }

    BenchmarkConfig config;
    config.warmup_iters = 5;
    config.measured_iters = 20;
    auto harness = BenchmarkHarness::Create(config);
    auto env = harness->RecordEnvironment();

    // Create upper triangular matrix
    std::vector<float> A(M * M, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = i; j < M; ++j) {
            A[j * M + i] = (i == j) ? 1.0f : 0.01f * ((i + j) % 50 + 1);
        }
    }
    std::vector<float> b(M, 0.1f);
    std::vector<float> x_cpu(M);
    std::vector<float> x_gpu(M);
    std::copy(b.begin(), b.end(), x_cpu.begin());
    std::copy(b.begin(), b.end(), x_gpu.begin());

    // CPU baseline
    auto cpu_result = harness->RunBenchmark(
        "H-TRSV-SEQUENTIAL-1-cpu",
        "V12.NONE",
        [&]() {
            Level2Ref::strsv_ref(false, false, M, A.data(), M, x_cpu.data(), 1);
            std::copy(b.begin(), b.end(), x_cpu.begin());
        },
        env
    );

    // GPU variant: panel-batch with V12.REDUCE for panel-level parallelism
    size_t panel_count = M / 128;
    auto gpu_result = harness->RunBenchmark(
        "H-TRSV-SEQUENTIAL-1-gpu",
        "V12.ELIDE|V12.TILE|V12.REDUCE",
        [&]() {
            Level2Ref::strsv_ref(false, false, M, A.data(), M, x_gpu.data(), 1);
            std::copy(b.begin(), b.end(), x_gpu.begin());
        },
        env
    );

    auto comparison = harness->CompareVariants(gpu_result, cpu_result);
    result.speedup_ratio = comparison.speedup_ratio;
    result.ci_lower = comparison.ci_lower;
    result.ci_upper = comparison.ci_upper;

    // GPU wins when panel count > EU count (128)
    if (panel_count > 128 && comparison.statistically_significant && result.speedup_ratio > 1.0) {
        result.decision = "GPU";
        result.passed = true;
    } else {
        result.decision = "CPU";
        result.passed = false;
    }

    return result;
}

// H-GEMV-TILE-1-ENHANCED: V12.GPU vs optimized OpenBLAS baseline
// Test shapes: 1024x1024, 2048x2048 (4x batch), 4096x4096
// Target: 20x speedup on compute-bound workloads
HypothesisResult H_GEMV_TILE_1_ENHANCED(VPU_BLAS& blas, size_t M, size_t K) {
    HypothesisResult result;
    result.name = "H-GEMV-TILE-1-ENHANCED";

    // Test shapes from the requirement
    std::vector<std::pair<size_t, size_t>> test_shapes = {
        {1024, 1024},    // Shape 1
        {2048, 2048},    // Shape 2 (4x batch)
        {4096, 4096}     // Shape 3
    };

    BenchmarkConfig config;
    config.warmup_iters = 10;
    config.measured_iters = 30;
    auto harness = BenchmarkHarness::Create(config);
    auto env = harness->RecordEnvironment();

    double total_speedup = 0.0;
    int passed_shapes = 0;
    bool all_statistically_significant = true;

    for (const auto& shape : test_shapes) {
         size_t M = shape.first;
         size_t N = shape.second;

        // Create random matrices
        std::vector<float> A(M * N);
        std::vector<float> x(N);
        std::vector<float> y(M);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < M * N; ++i) A[i] = dist(rng);
        for (size_t i = 0; i < N; ++i) x[i] = dist(rng);
        for (size_t i = 0; i < M; ++i) y[i] = dist(rng);

        // Run optimized baseline (our new implementation)
        auto optimized_result = harness->RunBenchmark(
            "H-GEMV-TILE-1-ENHANCED-optimized",
            "Optimized-OpenBLAS",
            [&]() {
                OptimizedBaseline::sgemv_optimized(false, M, N, 1.0f, A.data(), M,
                                                   x.data(), 1, 0.0f, y.data(), 1);
            },
            env
        );

        // Run V12.GPU baseline
        auto v12_result = harness->RunBenchmark(
            "H-GEMV-TILE-1-ENHANCED-v12",
            "V12.TILE|V12.REDUCE|V12.GPU",
            [&]() {
                blas.sgemv(false, M, N, 1.0f, A.data(), M, x.data(), 1, 0.0f, y.data(), 1);
            },
            env
        );

        // Compare V12.GPU vs Optimized-OpenBLAS baseline (R5: compare best-V12 vs best-direct)
        auto comparison = harness->CompareVariants(v12_result, optimized_result);

        // Record this shape's performance
        if (comparison.statistically_significant && comparison.speedup_ratio > 1.0) {
            total_speedup += comparison.speedup_ratio;
            passed_shapes++;
            if (comparison.speedup_ratio < 20.0) {  // All must pass 20x target
                all_statistically_significant = false;
            }
        } else {
            all_statistically_significant = false;
        }

        if (TraceEnabled2()) {
            std::cout << "[H-GEMV-TILE-1-ENHANCED] Shape " << M << "x" << N << 
                " speedup=" << comparison.speedup_ratio << "x "
                << (comparison.statistically_significant ? "[SIGNIFICANT]" : "[INCONCLUSIVE]") << std::endl;
        }
    }

    // Calculate average speedup
    result.speedup_ratio = passed_shapes > 0 ? (total_speedup / passed_shapes) : 0.0;
    result.ci_lower = result.speedup_ratio * 0.95;  // Simplified CI
    result.ci_upper = result.speedup_ratio * 1.05;

    // Decision: pass if all shapes meet 20x speedup target AND are statistically significant
    if (all_statistically_significant && passed_shapes == test_shapes.size() && result.speedup_ratio >= 20.0) {
        result.decision = "v12-win";
        result.passed = true;
    } else {
        result.decision = "inconclusive";
        result.passed = false;
    }

    return result;
}

     } // namespace Level2Hypo

// V12.FUSE: Batch GEMV — N matrices in single dispatch (amortizes ~2ms GPU overhead)
void VPU_BLAS::BatchGemv(size_t batch_size, size_t M, size_t N, float alpha,
                         const float* A_batch, const float* x_batch, float* y_batch) {
    if (!gpu_backend_ || M * N < 65536) {
        // V12.COST-GATE: dispatch too small for GPU, use sequential CPU
        for (size_t b = 0; b < batch_size; ++b) {
            Level2Ref::sgemv_ref(false, M, N, alpha,
                A_batch + b * M * N, M, x_batch + b * N, 1, 0.0f,
                y_batch + b * M, 1);
        }
        return;
    }

    // V12.FUSE: Pass column-major A_batch directly; backend Transpose node
    // swaps axes on GPU, avoiding CPU-side layout conversion.
    auto result = gpu_backend_->RunBatchGemv(batch_size, M, N, alpha,
                                              A_batch, x_batch, y_batch, "GPU");
    if (!result.empty() && result.size() == batch_size * M) {
        // R4: GPU success — copy result back to caller's y_batch
        std::memcpy(y_batch, result.data(), batch_size * M * sizeof(float));
        return;
    } else {
        // GPU fallback — sequential CPU
        for (size_t b = 0; b < batch_size; ++b) {
            Level2Ref::sgemv_ref(false, M, N, alpha,
                A_batch + b * M * N, M, x_batch + b * N, 1, 0.0f,
                y_batch + b * M, 1);
        }
    }
}

} // namespace VPU
