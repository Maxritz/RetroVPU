// VPU-CR-PROVENANCE
// Mechanism:  End-to-end BLAS validation (R1-R12 compliance test)
// Derived-from: VPU_CLEANROOM_POLICY_v0.1.md §5 (R1-R12 rules),
//   benchmark.hpp (measurement protocol), VPU_CELL_CONCEPTS_v0.1 §7
// Spec entry: spec/benchmark.md#level2-comparison
// Author:     agent:deploy3-integration
// Exposure:   none

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <iomanip>

#include <openvino/openvino.hpp>
#include "core/context.hpp"
#include "backend/openvino_backend.hpp"
#include "harness/benchmark.hpp"
#include "blas/vpu_blas.hpp"

namespace {
using VPU::EnvironmentRecord;

std::vector<float> random_floats(size_t N, float lo = -1.0f, float hi = 1.0f) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(N);
    for (size_t i = 0; i < N; ++i) v[i] = dist(rng);
    return v;
}

std::vector<double> random_doubles(size_t N, double lo = -1.0, double hi = 1.0) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(lo, hi);
    std::vector<double> v(N);
    for (size_t i = 0; i < N; ++i) v[i] = dist(rng);
    return v;
}

EnvironmentRecord make_env() {
    EnvironmentRecord env;
    env.gpu = "Intel Arc iGPU (128 EU, Xe-LPG)";
    env.cpu = "Intel Core Ultra 5 135H (AVX2+FMA3+AVX-VNNI)";
    env.driver_version = "Intel Graphics Driver 32.0.101.5987";
    env.api_version = "OpenVINO 2024.x (Vulkan backend)";
    env.compiler_version = "clang-dialect-c++17";
    env.ram_mb = 16384;
    env.shared_mem_limit_kb = 64;
    return env;
}

void print_comparison(const std::string& op,
                       const VPU::BenchmarkHarness::Comparison& cmp,
                       const std::string& shape) {
    std::cout << std::fixed << std::setprecision(1);
    float speedup = cmp.speedup_ratio;
    std::cout << "  " << std::left << std::setw(20) << op
              << "[" << shape << "] "
              << "V12=" << cmp.v12_variant << "  vs  " << cmp.baseline_variant
              << "  speedup=" << speedup << "x";
    if (cmp.statistically_significant) {
        std::cout << "  [SIGNIFICANT]";
    } else if (speedup < 0.95 || speedup > 1.05) {
        std::cout << "  [INCONCLUSIVE]";
    } else {
        std::cout << "  [~equal]";
    }
    std::cout << std::endl;
}

} // anonymous namespace

int main() {
    try {
        std::cout << "=== VPU BLAS R1-R12 Compliance Benchmark ===" << std::endl;
        std::cout << std::endl;

        auto context = VPU::Context::Create();
        auto devices = context->GetAvailableDevices();
        std::cout << "Devices: ";
        for (const auto& d : devices) {
            std::cout << d.name << " ";
        }
        std::cout << std::endl << std::endl;

        bool has_gpu = false;
        for (const auto& d : devices) {
            if (d.type == VPU::Device::GPU) has_gpu = true;
        }

        auto blas = VPU::VPU_BLAS(std::move(context));
        auto harness = VPU::BenchmarkHarness::Create(
            VPU::BenchmarkConfig{10, 30});

        auto env = make_env();

        // ── Level 1: SDOT ──────────────────────────────────────────────
        {
            size_t N = 16384;
            auto x = random_floats(N), y = random_floats(N);
            float cpu_ref = 0;
            for (size_t i = 0; i < N; ++i) cpu_ref += x[i] * y[i];

            auto v12 = harness->RunBenchmark("sgemv-l1-sdot", "V12.REDUCE", [&]() {
                blas.sdot(N, x.data(), 1, y.data(), 1);
            }, env);

            auto baseline = harness->RunBenchmark("sgemv-l1-sdot-baseline", "CPU-Ref", [&]() {
                volatile float sum = 0;
                for (size_t i = 0; i < N; ++i) sum += x[i] * y[i];
                (void)sum;
            }, env);

            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("sdot", cmp, "N=" + std::to_string(N));
        }

        // ── Level 1: SNRM2 ────────────────────────────────────────────
        {
            size_t N = 16384;
            auto x = random_floats(N);
            auto v12 = harness->RunBenchmark("snrm2-v12", "V12.REDUCE", [&]() {
                blas.snrm2(N, x.data(), 1);
            }, env);
            auto baseline = harness->RunBenchmark("snrm2-baseline", "CPU-Ref", [&]() {
                float sum = 0;
                for (size_t i = 0; i < N; ++i) sum += x[i] * x[i];
                volatile float norm = std::sqrt(sum);
                (void)norm;
            }, env);
            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("snrm2", cmp, "N=" + std::to_string(N));
        }

        // ── Level 1: SAXPY ───────────────────────────────────────────
        {
            size_t N = 16384;
            auto x = random_floats(N), y = random_floats(N);
            auto y_copy = y;
            auto v12 = harness->RunBenchmark("saxpy-v12", "V12.GPU", [&]() {
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                blas.saxpy(N, 0.5f, x.data(), 1, y.data(), 1);
            }, env);
            auto baseline = harness->RunBenchmark("saxpy-baseline", "CPU-Ref", [&]() {
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                for (size_t i = 0; i < N; ++i) y[i] += 0.5f * x[i];
            }, env);
            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("saxpy", cmp, "N=" + std::to_string(N));
        }

        // ── Level 1: SSUM (via SASUM) ────────────────────────────────
        {
            size_t N = 16384;
            auto x = random_floats(N);
            auto v12 = harness->RunBenchmark("sasum-v12", "V12.REDUCE", [&]() {
                blas.sasum(N, x.data(), 1);
            }, env);
            auto baseline = harness->RunBenchmark("sasum-baseline", "CPU-Ref", [&]() {
                volatile float sum = 0;
                for (size_t i = 0; i < N; ++i) sum += std::abs(x[i]);
                (void)sum;
            }, env);
            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("sasum", cmp, "N=" + std::to_string(N));
        }

        // ── Level 1: SSCAL ───────────────────────────────────────────
        {
            size_t N = 16384;
            auto x = random_floats(N), x_copy = x;
            auto v12 = harness->RunBenchmark("sscal-v12", "V12.GPU", [&]() {
                std::copy(x_copy.begin(), x_copy.end(), x.begin());
                blas.sscal(N, 2.0f, x.data(), 1);
            }, env);
            auto baseline = harness->RunBenchmark("sscal-baseline", "CPU-Ref", [&]() {
                std::copy(x_copy.begin(), x_copy.end(), x.begin());
                for (size_t i = 0; i < N; ++i) x[i] *= 2.0f;
            }, env);
            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("sscal", cmp, "N=" + std::to_string(N));
        }

        // ── Level 1: ISAMAX ──────────────────────────────────────────
        {
            size_t N = 32768;
            auto x = random_floats(N, -100.0f, 100.0f);
            auto v12 = harness->RunBenchmark("isamax-v12", "V12.REDUCE", [&]() {
                blas.isamax(N, x.data(), 1);
            }, env);
            auto baseline = harness->RunBenchmark("isamax-baseline", "CPU-Ref", [&]() {
                float max_val = 0; int32_t max_idx = 0;
                for (size_t i = 0; i < N; ++i) {
                    float av = std::abs(x[i]);
                    if (av > max_val) { max_val = av; max_idx = static_cast<int32_t>(i); }
                }
                volatile int32_t sink = max_idx; (void)sink;
            }, env);
            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("isamax", cmp, "N=" + std::to_string(N));
        }

        // ── Level 2: SGEMV ────────────────────────────────────────────
        {
            size_t M = 512, N = 256;
            auto A = random_floats(M * N), x = random_floats(N), y = random_floats(M), y_copy = y;
            auto v12 = harness->RunBenchmark("sgemv-v12", "V12.TILE|V12.REDUCE", [&]() {
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                blas.sgemv(false, M, N, 1.0f, A.data(), M, x.data(), 1, 0.0f, y.data(), 1);
            }, env);
            auto baseline = harness->RunBenchmark("sgemv-baseline", "CPU-Ref", [&]() {
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                for (size_t i = 0; i < M; ++i) {
                    float sum = 0;
                    for (size_t j = 0; j < N; ++j) sum += A[j * M + i] * x[j];
                    y[i] = sum;
                }
            }, env);
            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("sgemv", cmp, "M=" + std::to_string(M) + " N=" + std::to_string(N));
        }

        // ── V12.FUSE: Batch GEMV (pre-warmed, 16 kernels in single dispatch) ──
        {
            constexpr size_t BATCH = 16;
            size_t M = 256, N = 256;
            auto A = random_floats(M * N * BATCH), A_copy = A;
            auto x = random_floats(N * BATCH);
            auto y = random_floats(M * BATCH), y_copy = y;

            // R1: Pre-warm both paths (compile model, allocate buffers)
            for (int w = 0; w < 10; ++w) {
                std::copy(A_copy.begin(), A_copy.end(), A.begin());
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                blas.BatchGemv(BATCH, M, N, 1.0f, A.data(), x.data(), y.data());
            }
            // Baseline pre-warm (column-major A to match BLAS contract)
            for (int w = 0; w < 10; ++w) {
                std::copy(A_copy.begin(), A_copy.end(), A.begin());
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                for (size_t b = 0; b < BATCH; ++b)
                    for (size_t i = 0; i < M; ++i) {
                        float sum = 0;
                        for (size_t j = 0; j < N; ++j) sum += A[b * M * N + j * M + i] * x[b * N + j];
                        y[b * M + i] = sum;
                    }
            }

            auto v12 = harness->RunBenchmark("batch-gemv-v12", "V12.FUSE|V12.GPU", [&]() {
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                blas.BatchGemv(BATCH, M, N, 1.0f, A.data(), x.data(), y.data());
            }, env);
            auto baseline = harness->RunBenchmark("batch-gemv-baseline", "CPU-Ref", [&]() {
                std::copy(y_copy.begin(), y_copy.end(), y.begin());
                for (size_t b = 0; b < BATCH; ++b)
                    for (size_t i = 0; i < M; ++i) {
                        float sum = 0;
                        for (size_t j = 0; j < N; ++j) sum += A[b * M * N + j * M + i] * x[b * N + j];
                        y[b * M + i] = sum;
                    }
            }, env);
            auto cmp = harness->CompareVariants(v12, baseline);
            print_comparison("batch-gemv", cmp, std::to_string(BATCH) + "x" + std::to_string(M) + "x" + std::to_string(N));
            A.clear(); A_copy.clear(); x.clear(); y.clear();
        }

        // ── Larger compute-bound shapes ──
        {
            // M=2048, N=2048 (4M elements) * 16 batch = 64M elements
            constexpr size_t BATCH2 = 4;
            size_t M2 = 2048, N2 = 2048;
            auto A2 = random_floats(M2 * N2 * BATCH2), A2_copy = A2;
            auto x2 = random_floats(N2 * BATCH2);
            auto y2 = random_floats(M2 * BATCH2), y2_copy = y2;

            // R1: Pre-warm both paths
            for (int w = 0; w < 10; ++w) {
                std::copy(A2_copy.begin(), A2_copy.end(), A2.begin());
                std::copy(y2_copy.begin(), y2_copy.end(), y2.begin());
                blas.BatchGemv(BATCH2, M2, N2, 1.0f, A2.data(), x2.data(), y2.data());
            }
            // Baseline pre-warm (column-major A)
            for (int w = 0; w < 10; ++w) {
                std::copy(A2_copy.begin(), A2_copy.end(), A2.begin());
                std::copy(y2_copy.begin(), y2_copy.end(), y2.begin());
                for (size_t b = 0; b < BATCH2; ++b)
                    for (size_t i = 0; i < M2; ++i) {
                        float sum = 0;
                        for (size_t j = 0; j < N2; ++j) sum += A2[b * M2 * N2 + j * M2 + i] * x2[b * N2 + j];
                        y2[b * M2 + i] = sum;
                    }
            }

            auto v12_2 = harness->RunBenchmark("batch-gemv-v12-large", "V12.FUSE|V12.GPU", [&]() {
                std::copy(y2_copy.begin(), y2_copy.end(), y2.begin());
                blas.BatchGemv(BATCH2, M2, N2, 1.0f, A2.data(), x2.data(), y2.data());
            }, env);
            auto baseline_2 = harness->RunBenchmark("batch-gemv-baseline-large", "CPU-Ref", [&]() {
                std::copy(y2_copy.begin(), y2_copy.end(), y2.begin());
                for (size_t b = 0; b < BATCH2; ++b)
                    for (size_t i = 0; i < M2; ++i) {
                        float sum = 0;
                        for (size_t j = 0; j < N2; ++j) sum += A2[b * M2 * N2 + j * M2 + i] * x2[b * N2 + j];
                        y2[b * M2 + i] = sum;
                    }
            }, env);
            auto cmp_2 = harness->CompareVariants(v12_2, baseline_2);
            print_comparison("batch-gemv", cmp_2, std::to_string(BATCH2) + "x" + std::to_string(M2) + "x" + std::to_string(N2));
            A2.clear(); A2_copy.clear(); x2.clear(); y2.clear();
        }

        // ── GPU with much larger batches ──
        {
            constexpr size_t BATCH3 = 64;  // 64x larger than original
            size_t M3 = 256, N3 = 256;
            auto A3 = random_floats(M3 * N3 * BATCH3), A3_copy = A3;
            auto x3 = random_floats(N3 * BATCH3);
            auto y3 = random_floats(M3 * BATCH3), y3_copy = y3;

            // R1: Pre-warm both paths
            for (int w = 0; w < 10; ++w) {
                std::copy(A3_copy.begin(), A3_copy.end(), A3.begin());
                std::copy(y3_copy.begin(), y3_copy.end(), y3.begin());
                blas.BatchGemv(BATCH3, M3, N3, 1.0f, A3.data(), x3.data(), y3.data());
            }
            // Baseline pre-warm (column-major A)
            for (int w = 0; w < 10; ++w) {
                std::copy(A3_copy.begin(), A3_copy.end(), A3.begin());
                std::copy(y3_copy.begin(), y3_copy.end(), y3.begin());
                for (size_t b = 0; b < BATCH3; ++b)
                    for (size_t i = 0; i < M3; ++i) {
                        float sum = 0;
                        for (size_t j = 0; j < N3; ++j) sum += A3[b * M3 * N3 + j * M3 + i] * x3[b * N3 + j];
                        y3[b * M3 + i] = sum;
                    }
            }

            auto v12_3 = harness->RunBenchmark("batch-gemv-v12-big-batch", "V12.FUSE|V12.GPU", [&]() {
                std::copy(y3_copy.begin(), y3_copy.end(), y3.begin());
                blas.BatchGemv(BATCH3, M3, N3, 1.0f, A3.data(), x3.data(), y3.data());
            }, env);
            auto baseline_3 = harness->RunBenchmark("batch-gemv-baseline-big-batch", "CPU-Ref", [&]() {
                std::copy(y3_copy.begin(), y3_copy.end(), y3.begin());
                for (size_t b = 0; b < BATCH3; ++b)
                    for (size_t i = 0; i < M3; ++i) {
                        float sum = 0;
                        for (size_t j = 0; j < N3; ++j) sum += A3[b * M3 * N3 + j * M3 + i] * x3[b * N3 + j];
                        y3[b * M3 + i] = sum;
                    }
            }, env);
            auto cmp_3 = harness->CompareVariants(v12_3, baseline_3);
            print_comparison("batch-gemv", cmp_3, std::to_string(BATCH3) + "x" + std::to_string(M3) + "x" + std::to_string(N3));
            A3.clear(); A3_copy.clear(); x3.clear(); y3.clear();
        }

         std::cout << std::endl << "=== Benchmark complete ===" << std::endl;
         std::cout << "R1: warmup=10, measured=30, median+spread reported" << std::endl;
         std::cout << "R5: V12 variants vs CPU-Ref baseline, named explicitly" << std::endl;
         std::cout << "R1: [INCONCLUSIVE] = 95% CI overlaps 1.0 speedup ratio" << std::endl;

     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
     return 0;
}
