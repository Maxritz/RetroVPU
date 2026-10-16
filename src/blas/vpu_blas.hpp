// VPU-CR-PROVENANCE: author=vpu-foundation shard=A
// Clean-room origin: this project (V12-VPU clean-room).
// Frozen shared interface for the V12-VPU BLAS layer.
// All implementing shards include this header. No external code is copied.
//
// References: Lawson et al. (1979) §6 (BLAS), Golub & Van Loan (2013) §15 (Level-3),
//   Demmel (1990) blocked algorithm, VPU_CELL_CONCEPTS §5 (cell/hierarchy).

#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <array>
#include <algorithm>
#include <cmath>

namespace Vpu {

// ── R9: V12 method flags — independently switchable ablation axes ─────
// R12: tags label *intent* only; ablation results validate them.
enum MethodFlag : uint32_t {
    M_NONE       = 0,
    M_TILE       = 1u << 0,   // V12.TILE      — SLM 128×128×K tile
    M_FUSE       = 1u << 1,   // V12.FUSE      — N ops → 1 dispatch
    M_ELIDE      = 1u << 2,   // V12.ELIDE     — skip provably-inactive work
    M_PRECISION  = 1u << 3,   // V12.PRECISION — per-stage store/compute/accumulator
    M_COST_GATE  = 1u << 4,   // V12.COST-GATE — cost model selects rewrites
    M_REDUCE     = 1u << 5,   // V12.REDUCE    — subgroup-parallel reduction
    M_ALL        = M_TILE | M_FUSE | M_ELIDE | M_PRECISION | M_COST_GATE | M_REDUCE,
};
using MethodSet = uint32_t;

inline const char* method_str(MethodSet m) {
    if (m == M_NONE) return "M_NONE";
    static char buf[256];
    buf[0] = '\0';
    if (m & M_TILE)       std::strcat(buf, "TILE ");
    if (m & M_FUSE)       std::strcat(buf, "FUSE ");
    if (m & M_ELIDE)      std::strcat(buf, "ELIDE ");
    if (m & M_PRECISION)  std::strcat(buf, "PREC ");
    if (m & M_COST_GATE)  std::strcat(buf, "COST ");
    if (m & M_REDUCE)     std::strcat(buf, "REDUCE ");
    return buf;
}

// ── BLAS character-flag helpers (shared across translation units) ─────
inline bool isTransposed(char c) { return c == 'T' || c == 't' || c == 'C' || c == 'c'; }
inline bool isUpper(char c)      { return c == 'U' || c == 'u'; }
inline bool isUnitDiag(char c)   { return c == 'U' || c == 'u'; }
inline bool isLeftSide(char c)   { return c == 'L' || c == 'l'; }

// ── Backend selection ────────────────────────────────────────────────
enum class Backend : uint8_t { CPU, GPU, NPU, MULTI, AUTO };

inline const char* backend_str(Backend b) {
    switch (b) {
        case Backend::CPU:  return "CPU";
        case Backend::GPU:  return "GPU";
        case Backend::NPU:  return "NPU";
        case Backend::MULTI:return "MULTI";
        case Backend::AUTO: return "AUTO";
    }
    return "UNKNOWN";
}

// ── Data types ──────────────────────────────────────────────────────
enum class DType : uint8_t { F32, F64, I32, I8, U8, BF16, F16 };

inline size_t dtype_bytes(DType d) {
    switch (d) {
        case DType::F32:  return 4;
        case DType::F64:  return 8;
        case DType::I32:  return 4;
        case DType::I8:   return 1;
        case DType::U8:   return 1;
        case DType::BF16: return 2;
        case DType::F16:  return 2;
    }
    return 4;
}

// ── Memory layout ────────────────────────────────────────────────────
enum class Layout : uint8_t { ROW_MAJOR, COL_MAJOR };

// ── R3: Workload contract — machine-readable, must exist before timing ──
struct WorkloadContract {
    const char*  op;               // operation name
    int32_t      M, N, K;          // dimensions
    DType        dtype_a;
    DType        dtype_b;
    DType        dtype_c;          // store type
    DType        dtype_acc;        // accumulator type
    Layout       layout;
    int32_t      lda, ldb, ldc;    // leading dimensions
    float        tolerance;        // R4 validation tolerance
    int32_t      dispatch_count;   // number of dispatches
    int32_t      workgroup_m;      // GPU workgroup shape
    int32_t      workgroup_n;
    int32_t      workgroup_k;
    int32_t      slm_tile_m;       // SLM tile (128×128×K)
    int32_t      slm_tile_n;
    int32_t      slm_tile_k;
    int32_t      batch_count;      // for _batch ops
    int32_t      stride_a;         // for _strided ops
    int32_t      stride_b;
    int32_t      stride_c;
    int32_t      uplo;             // 0=upper, 1=lower (triangular/symmetric)
    int32_t      side;             // 0=left, 1=right
    int32_t      trans_a;          // 0=N, 1=T, 2=C
    int32_t      trans_b;
    int32_t      diag;             // 0=non-unit, 1=unit diagonal
    float        alpha;            // scalar alpha
    float        beta;             // scalar beta
    size_t       bytes_read;
    size_t       bytes_written;
    int32_t      warmup_iters;     // R1: warmup iterations
    int32_t      bench_iters;      // R1: measured iterations
    uint64_t     source_hash;      // hash of source / reference
    uint64_t     spirv_hash;       // hash of SPIR-V
    MethodSet    methods;          // V12 methods applied
    Backend      backend;
    const char*  baseline;         // R5: explicit baseline name
    const char*  gpu_spv;          // SPIR-V path for GPU
    const char*  ov_model;         // OpenVINO model path (Class B)
    bool         validate_cpu;     // R4: run CPU reference
    bool         verbose;
};

// ── R7: Three timing boundaries ──────────────────────────────────────
struct TimingTriple {
    double gpu_timestamp_us;   // GPU-timestamp execution (vkCmdWriteTimestamp)
    double host_us;            // host submission/recording
    double e2e_us;             // end-to-end (incl. batch-bound copies/waits)
    double gpu_timestamp_median;
    double gpu_timestamp_p90;
    double gpu_timestamp_cv;   // coefficient of variation
    uint32_t sample_count;
};

// ── R4: Correctness + completeness gate ──────────────────────────────
struct CorrectnessResult {
    bool    passed;              // within tolerance
    bool    complete;            // every output written
    double  max_error;
    double  rms_error;
    int32_t outputs_checked;
};

// ── R8: Environment metadata (every result) ────────────────────────────
struct EnvironmentMeta {
    const char* device_name;
    const char* driver_version;
    const char* vulkan_version;
    const char* compiler;
    const char* device_type;
    int32_t     workgroup_size_limit;
    int32_t     max_shared_mem_bytes;
    int32_t     max_compute_units;
    int32_t     subgroup_size;
    int32_t     timestamp_valid_bits;
    float       timestamp_period_ns;
    uint64_t    pipeline_hash;
    uint64_t    shader_hash;
};

// ── R1: Full result with 95% CI on the ratio ───────────────────────────
struct VpuResult {
    TimingTriple          timing;
    CorrectnessResult     correctness;
    EnvironmentMeta       env;
    const char*           baseline;          // R5
    double                speedup_ratio;     // baseline_time / candidate_time
    double                ci_lower;          // 95% CI on ratio
    double                ci_upper;
    bool                  is_inconclusive;   // CI overlaps 1.0
    MethodSet             methods_used;
    const char*           verdict;           // "v12-win"|"inconclusive"|"baseline-better"
};

// ── H-A: Host↔GPU staging ring (depth 1–3, contract-generated) ────────
struct StagingRing {
    int32_t  depth;                         // 1–3
    std::array<void*, 3> buffers{};
    std::array<size_t, 3> sizes{};
    Backend  backend;
    int32_t  current_slot;
};

// ── H-B: Timeline tags per transfer stream ────────────────────────────
enum class TimelineTag : uint8_t {
    TRANSFER_IN  = 0,
    COMPUTE      = 1,
    TRANSFER_OUT = 2,
    BARRIER      = 3,
};

// ── H-C: Dispatch order sweep ──────────────────────────────────────────
enum class DispatchOrder : uint8_t {
    HOST_FIRST       = 0,
    GPU_FIRST        = 1,
    INTERLEAVED      = 2,
    PIPELINE_OVERLAP = 3,
};

// ── H-F, H-H: Cell concepts ────────────────────────────────────────────
struct ComputeCell {
    int32_t lanes;           // H-F: execution lanes per SIMD
    int32_t register_bytes;  // H-H: register file per workgroup
    int32_t slm_bytes;       // H-H: shared-local memory per workgroup
    int32_t wave_size;       // H-H: subgroup/wave size
    int32_t max_wgs;         // max workgroup size
};

// ── SLM tile constants (128×128×K) ───────────────────────────────────
constexpr int32_t TILE_M_DEFAULT = 128;
constexpr int32_t TILE_N_DEFAULT = 128;
constexpr int32_t TILE_K_DEFAULT = 64;

// ── Timing protocol constants (R1) ─────────────────────────────────────
constexpr int32_t WARMUP_ITERS_DEFAULT  = 10;
constexpr int32_t BENCH_ITERS_DEFAULT    = 30;
constexpr int32_t CI_SAMPLES_REQUIRED    = 20;

// ── OpenVINO MatMul op (Class B) ──────────────────────────────────────
// Interface declared here; the OpenVINO integration shard provides the
// implementation. We call through a function pointer so the dispatch
// table stays backend-agnostic.
struct OpenVINO_MatMul {
    bool     transpose_a;
    bool     transpose_b;
    DType    dtype_a;
    DType    dtype_b;
    DType    dtype_c;
    DType    dtype_acc;
    int32_t  m, n, k;
    float    scale;          // INT8 dequant scale (V12.PRECISION)

    using ExecuteFn = VpuResult(*)(const OpenVINO_MatMul&,
                                   const void* a, const void* b, void* c,
                                   const WorkloadContract&);

    // Set by the OpenVINO integration shard at link time (Class B).
    // nullptr if OpenVINO backend is unavailable.
    static inline ExecuteFn execute_impl = nullptr;

    // Execute MatMul via OpenVINO (Class B).
    static VpuResult execute(const OpenVINO_MatMul& params,
                             const void* a, const void* b, void* c,
                             const WorkloadContract& contract) {
        if (execute_impl)
            return execute_impl(params, a, b, c, contract);
        // Fallback: zero-out output and return inconclusive
        VpuResult r{};
        r.correctness.passed   = false;
        r.correctness.complete = false;
        r.verdict              = "inconclusive";
        r.env.device_name      = "OpenVINO-unavailable";
        return r;
    }
};

// ── Dispatch table function signatures (R9: independently switchable) ─
using sgemm_fn                = VpuResult(*)(const WorkloadContract&, void* ctx);
using dgemm_fn                = sgemm_fn;
using sbgemm_fn               = sgemm_fn;
using ssymm_fn                = sgemm_fn;
using dsymm_fn                = sgemm_fn;
using ssyrk_fn                = sgemm_fn;
using dsyrk_fn                = sgemm_fn;
using ssyr2k_fn               = sgemm_fn;
using dsyr2k_fn               = sgemm_fn;
using strmm_fn                = sgemm_fn;
using dtrmm_fn                = sgemm_fn;
using strsm_fn                = sgemm_fn;
using dtrsm_fn                = sgemm_fn;
using sgemm_batch_fn          = sgemm_fn;
using sgemm_batch_strided_fn  = sgemm_fn;
using somatcopy_fn            = sgemm_fn;
using domatcopy_fn            = sgemm_fn;
using simatcopy_fn            = sgemm_fn;
using dimatcopy_fn            = sgemm_fn;
using geadd_fn                = sgemm_fn;

// ── Opaque handles for backends ───────────────────────────────────────
struct GpuEngine;       // forward-declared: vulkan_engine.hpp
struct OpenVINOEngine;  // forward-declared: OpenVINO shard provides
struct CpuBackend;      // forward-declared: CPU reference

// ── Dispatch context ───────────────────────────────────────────────────
struct DispatchContext {
    GpuEngine*       gpu;
    OpenVINOEngine*  ov;
    CpuBackend*      cpu;
    ComputeCell      cell;
    StagingRing      staging;
    DispatchOrder    dispatch_order;
    int32_t          warmup_iters;
    int32_t          bench_iters;
};

// ── R9: independent ablation switches ──────────────────────────────────
struct Level3Config {
    MethodSet  methods       = M_ALL;
    Backend    backend       = Backend::AUTO;
    bool       tile          = true;
    bool       fuse          = true;
    bool       elide         = true;
    bool       precision     = true;
    bool       reduce        = true;
    bool       cost_gate     = true;
    int32_t    warmup_iters  = WARMUP_ITERS_DEFAULT;
    int32_t    bench_iters   = BENCH_ITERS_DEFAULT;
    const char* baseline     = "CPU-Ref";
};

struct MemoryConfig {
    MethodSet  methods       = M_ALL;
    Backend    backend       = Backend::AUTO;
    bool       tile          = true;
    bool       fuse          = true;
    float      alpha          = 1.0f;
    int32_t    warmup_iters  = WARMUP_ITERS_DEFAULT;
    int32_t    bench_iters   = BENCH_ITERS_DEFAULT;
    const char* baseline     = "CPU-Ref";
};

} // namespace Vpu

// ──────────────────────────────────────────────────────────────────────
// VPU namespace: shared interface for Level 1 + Level 2 BLAS (Shard B)
// Includes the existing project type headers (Class B: SDK headers).
// ──────────────────────────────────────────────────────────────────────
#include "../core/context.hpp"
#include "../model/tensor.hpp"
#include "../harness/workload_contract.hpp"
#include "../harness/benchmark.hpp"
#include "../backend/openvino_backend.hpp"

namespace VPU {

// R9: V12 method bit-flags — independently switchable ablation axes
constexpr uint32_t V12_NONE      = 0x00;
constexpr uint32_t V12_TILE      = Vpu::M_TILE;
constexpr uint32_t V12_FUSE      = Vpu::M_FUSE;
constexpr uint32_t V12_ELIDE     = Vpu::M_ELIDE;
constexpr uint32_t V12_REDUCE    = Vpu::M_REDUCE;
constexpr uint32_t V12_PRECISION = Vpu::M_PRECISION;
constexpr uint32_t V12_COST_GATE = Vpu::M_COST_GATE;
constexpr uint32_t V12_ALL       = V12_TILE | V12_FUSE | V12_ELIDE | V12_REDUCE | V12_PRECISION | V12_COST_GATE;

using V12Method = uint32_t;

// R5: baseline stays independent — named explicitly
// CPU-Ref = hand-authored clean-room reference (R6), independent of V12 methods
inline const char* BASELINE_CPU_REF = "CPU-Ref";

// Device routing target (R5, R7)
enum class DeviceType { CPU, GPU, NPU, MULTI };

// BLAS operation classification for the cost model
enum class BLASType : uint8_t {
    // Level 1
    DOT, AXPY, SCAL, ROT, SWAP, COPY, ASUM, NRM2, AMAX,
    // Level 2 (dense)
    GEMV, GER, SYR, SYR2, SYMV, TRMV, TRSV,
    // Level 2 (banded)
    GBMV, SBMV, TBMV, TBSV,
    // Level 2 (packed)
    SPMV, SPK, TPMV, TPSV,
    // Level 3
    GEMM, SYMM, SYRK, SYR2K, TRMM, TRSM, GEMM_BATCH, GEMM_BATCH_STRIDED,
    // Memory
    SOMATCOPY, DOMATCOPY, SIMATCOPY, DIMATCOPY, GEADD
};

// DevicePlan: output of the cost model (R6, R10)
struct DevicePlan {
    DeviceType  target         = DeviceType::CPU;
    V12Method   methods        = V12_NONE;
    size_t      workgroup_size = 256;
    size_t      tile_m         = 8;
    size_t      tile_n         = 8;
    size_t      tile_k         = 8;
};

// V12CostModel: selects device + methods per (op, M, N, K, dtype)
// R6: compares best-V12 vs best-direct on equal terms
// R10: methods are cost-gated, not applied blindly
struct V12CostModel {
    static DevicePlan Route(BLASType op, size_t M, size_t N, size_t K, DType dt);
};

// R6: Direct baseline version constants
inline const char* BASELINE_OPTIMIZED_OPENBLAS = "Optimized-OpenBLAS";

// VPU_BLAS: main dispatch class for Level 1 + Level 2 BLAS
// All methods follow R3 (contract) → R6 (cost-gate) → R4 (validate) → R7/R8 (measure)
class VPU_BLAS {
public:
    explicit VPU_BLAS(std::shared_ptr<Context> ctx = nullptr)
        : context_(std::move(ctx)) {
        if (context_) {
            gpu_backend_ = OpenVINOBackend::Create(context_, "GPU");
        }
    }

    std::shared_ptr<Context> context_;
    std::unique_ptr<OpenVINOBackend> gpu_backend_;

    // ── Level 1 BLAS ──────────────────────────────────────────────────
    // DOT: dot product
    float  sdot(size_t N, const float* x, size_t incx, const float* y, size_t incy);
    double ddot(size_t N, const double* x, size_t incx, const double* y, size_t incy);

    // NRM2: Euclidean norm
    float  snrm2(size_t N, const float* x, size_t incx);
    double dnrm2(size_t N, const double* x, size_t incx);

    // ASUM: sum of absolute values
    float  sasum(size_t N, const float* x, size_t incx);
    double dasum(size_t N, const double* x, size_t incx);

    // AMAX: index of element with max absolute value
    int32_t isamax(size_t N, const float* x, size_t incx);
    int32_t idamax(size_t N, const double* x, size_t incx);

    // SCAL: x = alpha * x
    void sscal(size_t N, float alpha, float* x, size_t incx);
    void dscal(size_t N, double alpha, double* x, size_t incx);

    // AXPY: y = alpha * x + y
    void saxpy(size_t N, float alpha, const float* x, size_t incx,
               float* y, size_t incy);
    void daxpy(size_t N, double alpha, const double* x, size_t incx,
               double* y, size_t incy);

    // ROT: apply plane rotation
    void srot(size_t N, float* x, size_t incx, float* y, size_t incy,
              float c, float s);
    void drot(size_t N, double* x, size_t incx, double* y, size_t incy,
              double c, double s);

    // SWAP: swap x and y
    void sswap(size_t N, float* x, size_t incx, float* y, size_t incy);
    void dswap(size_t N, double* x, size_t incx, double* y, size_t incy);

    // COPY: y = x
    void scopy(size_t N, const float* x, size_t incx, float* y, size_t incy);
    void dcopy(size_t N, const double* x, size_t incx, double* y, size_t incy);

    // ── Level 2 BLAS ──────────────────────────────────────────────────
    // GEMV: y = alpha * op(A) * x + beta * y, A is M×N
    void sgemv(bool trans, size_t M, size_t N, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
    void dgemv(bool trans, size_t M, size_t N, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

    // GER: A = alpha * x * y^T + A, A is M×N
    void sger(size_t M, size_t N, float alpha,
              const float* x, size_t incx,
              const float* y, size_t incy,
              float* A, size_t lda);
    void dger(size_t M, size_t N, double alpha,
              const double* x, size_t incx,
              const double* y, size_t incy,
              double* A, size_t lda);

    // SYR: A = alpha * x * x^T + A (symmetric rank-1, upper triangle)
    void ssyr(size_t N, float alpha,
              const float* x, size_t incx,
              float* A, size_t lda);
    void dsyr(size_t N, double alpha,
              const double* x, size_t incx,
              double* A, size_t lda);

    // SYR2: A = alpha * x * y^T + alpha * y * x^T + A (symmetric rank-2)
    void ssyr2(size_t N, float alpha,
               const float* x, size_t incx,
               const float* y, size_t incy,
               float* A, size_t lda);
    void dsyr2(size_t N, double alpha,
               const double* x, size_t incx,
               const double* y, size_t incy,
               double* A, size_t lda);

    // SYMV: y = alpha * A * x + beta * y (symmetric matrix-vector)
    void ssymv(size_t N, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
    void dsymv(size_t N, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

    // TRMV: x = op(A) * x (triangular, upper triangle)
    void strmv(bool trans, bool unit_diag,
               size_t N, const float* A, size_t lda,
               float* x, size_t incx);
    void dtrmv(bool trans, bool unit_diag,
               size_t N, const double* A, size_t lda,
               double* x, size_t incx);

    // TRSV: solve A * x = b (triangular, upper triangle)
    void strsv(bool trans, bool unit_diag,
               size_t N, const float* A, size_t lda,
               float* x, size_t incx);
    void dtrsv(bool trans, bool unit_diag,
               size_t N, const double* A, size_t lda,
               double* x, size_t incx);

    // GBMV: y = alpha * op(A) * x + beta * y (banded)
    void sgbmv(bool trans, size_t M, size_t N, size_t kl, size_t ku,
               float alpha, const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
    void dgbmv(bool trans, size_t M, size_t N, size_t kl, size_t ku,
               double alpha, const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

    // SBMV: y = alpha * A * x + beta * y (symmetric banded)
    void ssbmv(size_t N, size_t k, float alpha,
               const float* A, size_t lda,
               const float* x, size_t incx,
               float beta, float* y, size_t incy);
    void dsbmv(size_t N, size_t k, double alpha,
               const double* A, size_t lda,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

    // TBMV: x = op(A) * x (triangular banded)
    void stbmv(bool trans, bool unit_diag,
               size_t N, size_t k,
               const float* A, size_t lda,
               float* x, size_t incx);
    void dtbmv(bool trans, bool unit_diag,
               size_t N, size_t k,
               const double* A, size_t lda,
               double* x, size_t incx);

    // TBSV: solve A * x = b (triangular banded)
    void stbsv(bool trans, bool unit_diag,
               size_t N, size_t k,
               const float* A, size_t lda,
               float* x, size_t incx);
    void dtbsv(bool trans, bool unit_diag,
               size_t N, size_t k,
               const double* A, size_t lda,
               double* x, size_t incx);

    // SPMV: y = alpha * A * x + beta * y (packed symmetric)
    void spmv(size_t N, float alpha,
              const float* AP,
              const float* x, size_t incx,
              float beta, float* y, size_t incy);
    void dspmv(size_t N, double alpha,
               const double* AP,
               const double* x, size_t incx,
               double beta, double* y, size_t incy);

    // SSPR: A = alpha * x * x^T + A (packed symmetric rank-1)
    void sspr(size_t N, float alpha,
              const float* x, size_t incx,
              float* AP);
    void dspr(size_t N, double alpha,
              const double* x, size_t incx,
              double* AP);

    // TPMV: x = op(A) * x (packed triangular)
    void stpmv(bool trans, bool unit_diag,
               size_t N, const float* AP,
               float* x, size_t incx);
    void dtpmv(bool trans, bool unit_diag,
               size_t N, const double* AP,
               double* x, size_t incx);

    // TPSV: solve A * x = b (packed triangular)
    void stpsv(bool trans, bool unit_diag,
               size_t N, const float* AP,
               float* x, size_t incx);
    void dtpsv(bool trans, bool unit_diag,
               size_t N, const double* AP,
               double* x, size_t incx);

    // Non-standard aliases (from OpenBLAS cblas_ wrappers)
    void spv(size_t N, float alpha,
             const float* AP,
             const float* x, size_t incx,
             float beta, float* y, size_t incy);
    void dpv(size_t N, double alpha,
             const double* AP,
             const double* x, size_t incx,
             double beta, double* y, size_t incy);
    void dpmv(size_t N, double alpha,
              const double* AP,
              const double* x, size_t incx,
              double beta, double* y, size_t incy);
     void stri(size_t N, const float* A, size_t lda,
               float* x, size_t incx);
     void dtri(size_t N, const double* A, size_t lda,
               double* x, size_t incx);

     // V12.FUSE: Batch GEMV — N matrices in single dispatch (amortizes ~2ms GPU overhead)
     void BatchGemv(size_t batch_size, size_t M, size_t N, float alpha,
                    const float* A_batch, const float* x_batch, float* y_batch);
};

} // namespace VPU
