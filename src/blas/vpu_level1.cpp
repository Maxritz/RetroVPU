// VPU-CR-PROVENANCE
// Mechanism:  V12.REDUCE + V12.PRECISION + V12.COST-GATE (Level 1 BLAS)
// Derived-from: Lawson et al. (1979) §5-6 (BLAS), Golub & Van Loan (2013) §2
//   (vector operations, norms); VPU_CELL_CONCEPTS_v0.1 §2 (SLM),
//   §5.4 (H-F budget), §5.3 (H-C placement), §5.5 (H-E claim),
//   §6 (H-G alignment); VPU_CLEANROOM_POLICY_v0.1 §5 (method catalogue)
// Spec entry: spec/blas.md#level1
// Author:     agent:deploy3-shardA-retry
// Exposure:   none

#include "vpu_blas.hpp"
#include "../backend/openvino_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace VPU {

// R11: Sherlock-It perf tracing
using Clock = std::chrono::high_resolution_clock;
inline double us_elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
inline bool TraceEnabled() {
    static bool enabled = std::getenv("VPU_TRACE") != nullptr;
    return enabled;
}
inline void TRACE(const char* op, const char* tag, double elapsed_us, bool gpu_used) {
    if (!TraceEnabled()) return;
    std::cout << "[SHERLOCK] " << op << " " << tag << " "
              << elapsed_us << "us" << (gpu_used ? " [GPU]" : " [CPU]") << std::endl;
}
inline double us_elapsed_cond(Clock::time_point start) {
    return TraceEnabled() ? us_elapsed(start) : 0.0;
}

// ══════════════════════════════════════════════════════════════════════=
//  CPU reference implementations (R4: correctness gate)
//  Clean-room: Lawson et al. (1979) §5-6; Golub & Van Loan (2013) §2
// ══════════════════════════════════════════════════════════════════════

namespace Level1Ref {

float sdot_ref(size_t N, const float* x, size_t incx,
               const float* y, size_t incy) {
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i)
        sum += x[i * incx] * y[i * incy];
    return sum;
}

double ddot_ref(size_t N, const double* x, size_t incx,
                const double* y, size_t incy) {
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i)
        sum += x[i * incx] * y[i * incy];
    return sum;
}

float snrm2_ref(size_t N, const float* x, size_t incx) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < N; ++i)
        sum_sq += x[i * incx] * x[i * incx];
    return std::sqrt(sum_sq);
}

double dnrm2_ref(size_t N, const double* x, size_t incx) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < N; ++i)
        sum_sq += x[i * incx] * x[i * incx];
    return std::sqrt(sum_sq);
}

float sasum_ref(size_t N, const float* x, size_t incx) {
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i)
        sum += std::abs(x[i * incx]);
    return sum;
}

double dasum_ref(size_t N, const double* x, size_t incx) {
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i)
        sum += std::abs(x[i * incx]);
    return sum;
}

int32_t isamax_ref(size_t N, const float* x, size_t incx) {
    if (N == 0) return 0;
    float max_val = std::abs(x[0]);
    int32_t max_idx = 0;
    for (size_t i = 1; i < N; ++i) {
        float val = std::abs(x[i * incx]);
        if (val > max_val) {
            max_val = val;
            max_idx = static_cast<int32_t>(i);
        }
    }
    return max_idx;
}

int32_t idamax_ref(size_t N, const double* x, size_t incx) {
    if (N == 0) return 0;
    double max_val = std::abs(x[0]);
    int32_t max_idx = 0;
    for (size_t i = 1; i < N; ++i) {
        double val = std::abs(x[i * incx]);
        if (val > max_val) {
            max_val = val;
            max_idx = static_cast<int32_t>(i);
        }
    }
    return max_idx;
}

void sscal_ref(size_t N, float alpha, float* x, size_t incx) {
    for (size_t i = 0; i < N; ++i)
        x[i * incx] *= alpha;
}

void dscal_ref(size_t N, double alpha, double* x, size_t incx) {
    for (size_t i = 0; i < N; ++i)
        x[i * incx] *= alpha;
}

void saxpy_ref(size_t N, float alpha, const float* x, size_t incx,
               float* y, size_t incy) {
    for (size_t i = 0; i < N; ++i)
        y[i * incy] += alpha * x[i * incx];
}

void daxpy_ref(size_t N, double alpha, const double* x, size_t incx,
               double* y, size_t incy) {
    for (size_t i = 0; i < N; ++i)
        y[i * incy] += alpha * x[i * incx];
}

void srot_ref(size_t N, float* x, size_t incx, float* y, size_t incy,
              float c, float s) {
    for (size_t i = 0; i < N; ++i) {
        float xi = x[i * incx];
        float yi = y[i * incy];
        x[i * incx] = c * xi + s * yi;
        y[i * incy] = c * yi - s * xi;
    }
}

void drot_ref(size_t N, double* x, size_t incx, double* y, size_t incy,
              double c, double s) {
    for (size_t i = 0; i < N; ++i) {
        double xi = x[i * incx];
        double yi = y[i * incy];
        x[i * incx] = c * xi + s * yi;
        y[i * incy] = c * yi - s * xi;
    }
}

void sswap_ref(size_t N, float* x, size_t incx, float* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        float tmp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = tmp;
    }
}

void dswap_ref(size_t N, double* x, size_t incx, double* y, size_t incy) {
    for (size_t i = 0; i < N; ++i) {
        double tmp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = tmp;
    }
}

void scopy_ref(size_t N, const float* x, size_t incx, float* y, size_t incy) {
    for (size_t i = 0; i < N; ++i)
        y[i * incy] = x[i * incx];
}

void dcopy_ref(size_t N, const double* x, size_t incx, double* y, size_t incy) {
    for (size_t i = 0; i < N; ++i)
        y[i * incy] = x[i * incx];
}

} // namespace Level1Ref

// ══════════════════════════════════════════════════════════════════════
//  V12CostModel::Route — selects device + V12 methods per shape
//  R6: compare best-V12 vs best-direct, per shape, on equal terms
//  R10: methods are cost-gated, not applied blindly
// ══════════════════════════════════════════════════════════════════════

DevicePlan V12CostModel::Route(BLASType op, size_t M, size_t N, size_t K, DType dt) {
    DevicePlan plan;
    size_t total_elements = M * N * K;

    switch (op) {
    case BLASType::DOT:
    case BLASType::ASUM:
    case BLASType::NRM2:
        // SHERLOCK-TRAP-1: GPU dispatch overhead ~2ms; Level 1 never wins
        // without batching (V12.FUSE). Route to CPU until batch mode is implemented.
        plan.target = DeviceType::CPU;
        plan.methods = V12_NONE;
        break;

    case BLASType::AXPY:
    case BLASType::SCAL:
    case BLASType::COPY:
    case BLASType::SWAP:
    case BLASType::ROT:
        // H-L1-TILE-1: memory-bound; GPU wins when vector length > 16K
        if (M > 16384) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_TILE;
            plan.workgroup_size = 256;
            plan.tile_m = 256;
            plan.tile_n = 1;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::AMAX:
        // SHERLOCK-TRAP-1: RunAmax computes max|x| on GPU then CPU-scans for index.
        // GPU overhead (~2ms) + CPU scan makes GPU always slower. Use CPU.
        plan.target = DeviceType::CPU;
        plan.methods = V12_NONE;
        break;

    case BLASType::GEMV:
        // H-GEMV-TILE-1: SLM tile saturates L2 bandwidth; dispatch overhead
        // dominates below 256 outputs. GPU wins when M >= 512.
        if (M >= 512 && N >= 64) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_TILE | V12_REDUCE;
            plan.workgroup_size = 256;
            plan.tile_m = 256;
            plan.tile_n = 64;
            plan.tile_k = static_cast<size_t>(N);
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::GER:
        // SHERLOCK-TRAP-7: GPU dispatch overhead ~2000us; gate behind large M*N
        if (M * N > 1048576) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_TILE;
            plan.workgroup_size = 256;
            plan.tile_m = 32;
            plan.tile_n = 8;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::SYR:
    case BLASType::SYR2:
        // H-L2-SYR-1: triangle compute saves ~50%; GPU wins at N >= 512
        if (M >= 512) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_ELIDE | V12_TILE;
            plan.workgroup_size = 256;
            plan.tile_m = 32;
            plan.tile_n = 8;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::SYMV:
        // H-L2-SYMV-1: V12.ELIDE for symmetric matrix; GPU at N >= 512
        if (M >= 512) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_ELIDE | V12_TILE | V12_REDUCE;
            plan.workgroup_size = 256;
            plan.tile_m = 256;
            plan.tile_n = 64;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::TRMV:
        // H-L2-TRMV-1: upper-bounded; GPU at N >= 1024
        if (M >= 1024) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_TILE;
            plan.workgroup_size = 256;
            plan.tile_m = 32;
            plan.tile_n = 8;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::TRSV:
        // H-L2-TRSV-1: forward substitution sequential; GPU panel-count > 128
        if (M > 128 * 128) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_ELIDE | V12_TILE | V12_REDUCE;
            plan.workgroup_size = 256;
            plan.tile_m = 128;
            plan.tile_n = 128;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::GBMV:
    case BLASType::SBMV:
        // H-L2-BAND-1: banded ops; GPU at M*N > 8K (bandwidth-bound)
        if (M * N > 8192) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_TILE;
            plan.workgroup_size = 256;
            plan.tile_m = 32;
            plan.tile_n = 8;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::TBMV:
        if (M >= 512) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_TILE;
            plan.workgroup_size = 256;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::TBSV:
        if (M > 128 * 128) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_ELIDE | V12_TILE | V12_REDUCE;
            plan.workgroup_size = 256;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::SPMV:
    case BLASType::SPK:
        if (M >= 2048) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_ELIDE | V12_TILE | V12_REDUCE;
            plan.workgroup_size = 256;
            plan.tile_m = 256;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::TPMV:
        if (M >= 512) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_TILE;
            plan.workgroup_size = 256;
        }
        else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::TPSV:
        if (M > 128 * 128) {
            plan.target = DeviceType::GPU;
            plan.methods = V12_ELIDE | V12_TILE | V12_REDUCE;
            plan.workgroup_size = 256;
        } else {
            plan.target = DeviceType::CPU;
            plan.methods = V12_NONE;
        }
        break;

    case BLASType::GEMM:
    case BLASType::SYMM:
    case BLASType::SYRK:
    case BLASType::SYR2K:
    case BLASType::TRMM:
    case BLASType::TRSM:
        // Level 3 routed by Shard C's Vpu namespace; here just CPU fallback
        plan.target = DeviceType::CPU;
        plan.methods = V12_NONE;
        break;

    default:
        plan.target = DeviceType::CPU;
        plan.methods = V12_NONE;
        break;
    }

    return plan;
}

// ══════════════════════════════════════════════════════════════════════
//  VPU_BLAS Level 1 member function implementations
//  Each function: contract (R3) → route (V12.COST-GATE) →
//    validate (R4) → execute → record (R8)
// ══════════════════════════════════════════════════════════════════════

// ── DOT: dot product ──────────────────────────────────────────────────
float VPU_BLAS::sdot(size_t N, const float* x, size_t incx,
                     const float* y, size_t incy) {
    auto plan = V12CostModel::Route(BLASType::DOT, N, 1, 1, DType::FP32);
    auto _t0 = TraceEnabled() ? Clock::now() : Clock::time_point{};

    if (plan.target == DeviceType::CPU) {
        auto r = Level1Ref::sdot_ref(N, x, incx, y, incy);
        if (TraceEnabled()) TRACE("sdot", "V12.CPU", us_elapsed(_t0), false);
        return r;
    }
    if (gpu_backend_) {
        auto result = gpu_backend_->RunDot(N, x, y, "GPU");
        if (!result.empty()) {
            float ref = Level1Ref::sdot_ref(N, x, incx, y, incy);
            float max_err = std::abs(result[0] - ref);
            if (max_err > 1e-4f) {
                auto r = Level1Ref::sdot_ref(N, x, incx, y, incy);
                if (TraceEnabled()) TRACE("sdot", "V12.FB", us_elapsed(_t0), false);
                return r;
            }
            if (TraceEnabled()) TRACE("sdot", "V12.GPU", us_elapsed(_t0), true);
            return result[0];
        }
    }
    auto r = Level1Ref::sdot_ref(N, x, incx, y, incy);
    if (TraceEnabled()) TRACE("sdot", "V12.FB", us_elapsed(_t0), false);
    return r;
}

double VPU_BLAS::ddot(size_t N, const double* x, size_t incx,
                      const double* y, size_t incy) {
    auto plan = V12CostModel::Route(BLASType::DOT, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        return Level1Ref::ddot_ref(N, x, incx, y, incy);
    }
    // TODO: V12.GPU for FP64 — OpenVINO GPU may not support FP64; fallback to CPU
    double result = Level1Ref::ddot_ref(N, x, incx, y, incy);
    return result;
}

// ── NRM2: Euclidean norm ──────────────────────────────────────────────
float VPU_BLAS::snrm2(size_t N, const float* x, size_t incx) {
    auto _t0 = Clock::now();
    auto plan = V12CostModel::Route(BLASType::NRM2, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        auto r = Level1Ref::snrm2_ref(N, x, incx);
        TRACE("snrm2", "V12.CPU", us_elapsed(_t0), false);
        return r;
    }
    if (gpu_backend_) {
        std::vector<float> x_sq(N * incx);
        for (size_t i = 0; i < N; ++i) {
            float val = x[i * incx];
            x_sq[i * incx] = val * val;
        }
        auto result = gpu_backend_->RunReduceSum(N, x_sq.data(), "GPU");
        if (!result.empty()) {
            float ref = Level1Ref::snrm2_ref(N, x, incx);
            float ref_sq = ref * ref;
            float gpu_sq = result[0];
            if (std::abs(gpu_sq - ref_sq) <= 1e-3f * ref_sq) {
                auto r = static_cast<float>(std::sqrt(gpu_sq));
                TRACE("snrm2", "V12.GPU", us_elapsed(_t0), true);
                return r;
            }
        }
    }
    auto r = Level1Ref::snrm2_ref(N, x, incx);
    TRACE("snrm2", "V12.FB", us_elapsed(_t0), false);
    return r;
}

double VPU_BLAS::dnrm2(size_t N, const double* x, size_t incx) {
    auto plan = V12CostModel::Route(BLASType::NRM2, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        return Level1Ref::dnrm2_ref(N, x, incx);
    }
    double sum_sq = Level1Ref::dnrm2_ref(N, x, incx);
    return sum_sq;
}

// ── ASUM: sum of absolute values ───────────────────────────────────────
float VPU_BLAS::sasum(size_t N, const float* x, size_t incx) {
    auto _t0 = Clock::now();
    auto plan = V12CostModel::Route(BLASType::ASUM, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        auto r = Level1Ref::sasum_ref(N, x, incx);
        TRACE("sasum", "V12.CPU", us_elapsed(_t0), false);
        return r;
    }
    if (gpu_backend_) {
        std::vector<float> abs_x(N);
        for (size_t i = 0; i < N; ++i) abs_x[i] = std::abs(x[i * incx]);
        auto result = gpu_backend_->RunReduceSum(N, abs_x.data(), "GPU");
        if (!result.empty()) {
            TRACE("sasum", "V12.GPU", us_elapsed(_t0), true);
            return result[0];
        }
    }
    auto r = Level1Ref::sasum_ref(N, x, incx);
    TRACE("sasum", "V12.FB", us_elapsed(_t0), false);
    return r;
}

double VPU_BLAS::dasum(size_t N, const double* x, size_t incx) {
    auto plan = V12CostModel::Route(BLASType::ASUM, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        return Level1Ref::dasum_ref(N, x, incx);
    }
    return Level1Ref::dasum_ref(N, x, incx);
}

// ── AMAX: index of max absolute value ─────────────────────────────────
int32_t VPU_BLAS::isamax(size_t N, const float* x, size_t incx) {
    auto plan = V12CostModel::Route(BLASType::AMAX, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        // Inline for compiler optimization (avoids function call overhead vs baseline)
        if (N == 0) return 0;
        float max_val = std::abs(x[0]);
        int32_t max_idx = 0;
        for (size_t i = 1; i < N; ++i) {
            float val = std::abs(x[i * incx]);
            if (val > max_val) { max_val = val; max_idx = static_cast<int32_t>(i); }
        }
        return max_idx;
    }
    if (gpu_backend_) {
        auto result = gpu_backend_->RunAmax(N, x, "GPU");
        if (!result.empty()) {
            int32_t idx = static_cast<int32_t>(result[0]);
            if (idx >= 0 && idx < static_cast<int32_t>(N)) {
                return idx;
            }
        }
    }
    auto r = Level1Ref::isamax_ref(N, x, incx);
    return r;
}

int32_t VPU_BLAS::idamax(size_t N, const double* x, size_t incx) {
    auto plan = V12CostModel::Route(BLASType::AMAX, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        return Level1Ref::idamax_ref(N, x, incx);
    }
    return Level1Ref::idamax_ref(N, x, incx);
}

// ── SCAL: x = alpha * x ────────────────────────────────────────────────
void VPU_BLAS::sscal(size_t N, float alpha, float* x, size_t incx) {
    auto _t0 = Clock::now();
    auto plan = V12CostModel::Route(BLASType::SCAL, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::sscal_ref(N, alpha, x, incx);
        TRACE("sscal", "V12.CPU", us_elapsed(_t0), false);
        return;
    }
    if (gpu_backend_) {
        std::vector<float> x_vec(N);
        for (size_t i = 0; i < N; ++i) x_vec[i] = x[i * incx];
        auto result = gpu_backend_->RunScal(N, alpha, x_vec.data(), "GPU");
        if (!result.empty()) {
            for (size_t i = 0; i < N; ++i) x[i * incx] = result[i];
            TRACE("sscal", "V12.GPU", us_elapsed(_t0), true);
            return;
        }
    }
    Level1Ref::sscal_ref(N, alpha, x, incx);
    TRACE("sscal", "V12.FB", us_elapsed(_t0), false);
}

void VPU_BLAS::dscal(size_t N, double alpha, double* x, size_t incx) {
    auto plan = V12CostModel::Route(BLASType::SCAL, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::dscal_ref(N, alpha, x, incx);
    } else {
        Level1Ref::dscal_ref(N, alpha, x, incx);
    }
}

// ── AXPY: y = alpha * x + y ────────────────────────────────────────────
void VPU_BLAS::saxpy(size_t N, float alpha, const float* x, size_t incx,
                     float* y, size_t incy) {
    auto _t0 = Clock::now();
    auto plan = V12CostModel::Route(BLASType::AXPY, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::saxpy_ref(N, alpha, x, incx, y, incy);
        TRACE("saxpy", "V12.CPU", us_elapsed(_t0), false);
        return;
    }
    if (gpu_backend_) {
        std::vector<float> x_full(N), y_full(N);
        for (size_t i = 0; i < N; ++i) {
            x_full[i] = x[i * incx];
            y_full[i] = y[i * incy];
        }
        auto result = gpu_backend_->RunAxpy(N, alpha, x_full.data(), y_full.data(), "GPU");
        if (!result.empty()) {
            for (size_t i = 0; i < N; ++i) y[i * incy] = result[i];
            TRACE("saxpy", "V12.GPU", us_elapsed(_t0), true);
            return;
        }
    }
    Level1Ref::saxpy_ref(N, alpha, x, incx, y, incy);
    TRACE("saxpy", "V12.FB", us_elapsed(_t0), false);
}

void VPU_BLAS::daxpy(size_t N, double alpha, const double* x, size_t incx,
                     double* y, size_t incy) {
    auto plan = V12CostModel::Route(BLASType::AXPY, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::daxpy_ref(N, alpha, x, incx, y, incy);
    } else {
        Level1Ref::daxpy_ref(N, alpha, x, incx, y, incy);
    }
}

// ── ROT: plane rotation ────────────────────────────────────────────────
void VPU_BLAS::srot(size_t N, float* x, size_t incx, float* y, size_t incy,
                    float c, float s) {
    auto plan = V12CostModel::Route(BLASType::ROT, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::srot_ref(N, x, incx, y, incy, c, s);
    } else {
        Level1Ref::srot_ref(N, x, incx, y, incy, c, s);
    }
}

void VPU_BLAS::drot(size_t N, double* x, size_t incx, double* y, size_t incy,
                     double c, double s) {
    auto plan = V12CostModel::Route(BLASType::ROT, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::drot_ref(N, x, incx, y, incy, c, s);
    } else {
        Level1Ref::drot_ref(N, x, incx, y, incy, c, s);
    }
}

// ── SWAP: swap vectors ─────────────────────────────────────────────────
void VPU_BLAS::sswap(size_t N, float* x, size_t incx, float* y, size_t incy) {
    auto plan = V12CostModel::Route(BLASType::SWAP, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::sswap_ref(N, x, incx, y, incy);
    } else {
        Level1Ref::sswap_ref(N, x, incx, y, incy);
    }
}

void VPU_BLAS::dswap(size_t N, double* x, size_t incx, double* y, size_t incy) {
    auto plan = V12CostModel::Route(BLASType::SWAP, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::dswap_ref(N, x, incx, y, incy);
    } else {
        Level1Ref::dswap_ref(N, x, incx, y, incy);
    }
}

// ── COPY: y = x ────────────────────────────────────────────────────────
void VPU_BLAS::scopy(size_t N, const float* x, size_t incx, float* y, size_t incy) {
    auto plan = V12CostModel::Route(BLASType::COPY, N, 1, 1, DType::FP32);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::scopy_ref(N, x, incx, y, incy);
    } else {
        Level1Ref::scopy_ref(N, x, incx, y, incy);
    }
}

void VPU_BLAS::dcopy(size_t N, const double* x, size_t incx, double* y, size_t incy) {
    auto plan = V12CostModel::Route(BLASType::COPY, N, 1, 1, DType::FP64);

    if (plan.target == DeviceType::CPU) {
        Level1Ref::dcopy_ref(N, x, incx, y, incy);
    } else {
        Level1Ref::dcopy_ref(N, x, incx, y, incy);
    }
}

} // namespace VPU
