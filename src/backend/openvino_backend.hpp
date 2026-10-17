// VPU-CR-PROVENANCE
// Mechanism:  P3 staged local store (OpenVINO backend context)
// Derived-from: OpenVINO C++ API documentation (vendor specification, Class B)
// V12.VPU.Cell concepts applied:
//   - P3 staged local store: Compiled model cache reuses GPU kernels (avoids recompilation overhead)
//   - Deep command FIFO: V12.FUSE batches N ops into 1 dispatch via OpenVINO graph optimization
//   - On-board buffer I/O: Direct GPU buffer reads/writes, no host round-trips in steady-state
//   - Per-pixel lighting (DOT4): MatMul-based dot product reduction (V12.REDUCE) 
// Spec entry: spec/backend.md#openvino
// Author:     agent:claude@opencodo-vpu-session-001
// Exposure:   none

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <openvino/openvino.hpp>
#include "context.hpp"
#include "tensor.hpp"

namespace VPU {

// V12.VPU.Cell adaptation for OpenVINO:
// - P3 staged local store: Model compilation cached, reused across calls (avoids recompile)
// - Deep command FIFO: V12.FUSE via OpenVINO graph fusion (single dispatch for batched ops)
// - Direct buffer I/O: ov::Tensor maps GPU memory, no host round-trip in steady-state
// - DOT4-style compute: MatMul ops implement V12.REDUCE (per-element dot products)
class OpenVINOBackend {
public:
    static std::unique_ptr<OpenVINOBackend> Create(std::shared_ptr<Context> context,
                                                    const std::string& device_name);
    
    std::unique_ptr<Tensor> RunInference(const std::string& model_path, 
                                          const std::vector<std::unique_ptr<Tensor>>& inputs);
    
    void CompileModel(const std::string& model_path);
    
    const DeviceInfo& device_info() const { return device_info_; }
    
    // V12 GPU dispatch methods - execute V12 methods on GPU (OpenVINO)
    // V12.REDUCE: Dot product (sum of element-wise products)
    std::vector<float> RunDot(size_t N, const float* x, const float* y, const std::string& device = "GPU", size_t work_group_size = 256);
    
    // V12.REDUCE: Subgroup parallel reduction
    std::vector<float> RunReduceSum(size_t N, const float* data, const std::string& device = "GPU", size_t work_group_size = 256);
    
    // V12.REDUCE: Find maximum absolute value (used in ASUM, NRM2)
    std::vector<float> RunAmax(size_t N, const float* x, const std::string& device = "GPU", size_t work_group_size = 256);
    
    // V12.FUSE: Fused Scale (x = alpha * x)
    std::vector<float> RunScal(size_t N, float alpha, const float* x, const std::string& device = "GPU", size_t work_group_size = 256);
    
    // V12.FUSE: Fused AXPY (y = alpha * x + y)
    std::vector<float> RunAxpy(size_t N, float alpha, const float* x, const float* y, 
                               const std::string& device = "GPU", size_t work_group_size = 256);
    
    // V12.TILE + V12.REDUCE + V12.FUSE: Gemv (General Matrix-Vector Multiply)
    // Returns result vector (empty on failure → triggers CPU fallback)
    std::vector<float> RunGemv(bool trans, size_t M, size_t N, float alpha, float beta,
                               const float* A, const float* x, float* y_out, const std::string& device = "GPU");
    
    // V12.TILE + V12.REDUCE + V12.FUSE: Gemv FP64 (double precision)
    // Returns result vector (empty on failure → triggers CPU fallback)
    std::vector<double> RunGemvFP64(bool trans, size_t M, size_t N, double alpha, double beta,
                                    const double* A, const double* x, double* y_out, const std::string& device = "GPU");
    
    // V12.FUSE: Ger (General Rank-1 update: A = alpha * x * y^T + A)
    std::vector<float> RunGer(size_t M, size_t N, float alpha, const float* x,
                              const float* y, float* A_out, const std::string& device = "GPU");
    
    // V12.TILE + V12.REDUCE: Batch GEMV (batched matrix-vector multiply)
    std::vector<float> RunBatchGemv(size_t batch_size, size_t M, size_t N, float alpha,
                                    const float* A_batch, const float* x_batch,
                                    float* y_batch, const std::string& device = "GPU");

    // Timing breakdown (R7) - returned via output parameters
    struct ExecutionTimings {
        double gpu_time_us = 0.0;      // GPU-timestamp execution
        double host_time_us = 0.0;     // host submission/recording
        double e2e_time_us = 0.0;      // end-to-end
    };
    
    // Get timings from last execution (R7)
    ExecutionTimings GetLastTimings() const { return last_timings_; }
    
    // Device info accessors (R8)
    std::string GetDeviceName() const { return device_name_; }
    std::string GetDriverVersion() const;
    std::string GetApiVersion() const;
    
private:
    OpenVINOBackend(std::shared_ptr<Context> context, 
                    const std::string& device_name,
                    DeviceInfo device_info);
    
    // Helper: Create a reusable compiled model cache key
    std::string CreateCacheKey(const std::string& op_type, size_t M, size_t N, size_t K) const;
    
    // Helper: Get or compile a model for specific operation
    ov::CompiledModel GetOrCreateMatMulModel(size_t M, size_t N, size_t K);
    
    // Helper: Copy host data to OpenVINO tensor
    void FillTensor(ov::Tensor& tensor, const float* data, size_t size);
    void FillTensor(ov::Tensor& tensor, const double* data, size_t size);
    
    // Helper: Copy OpenVINO tensor data to host
    void CopyToHost(const ov::Tensor& tensor, float* data, size_t size);
    void CopyToHost(const ov::Tensor& tensor, double* data, size_t size);
    
    std::shared_ptr<Context> context_;
    ov::Core core_;  // Cached core for model compilation
    std::string device_name_;
    DeviceInfo device_info_;
    ExecutionTimings last_timings_;
    
    // Compiled model cache (for reuse across batches - V12.FUSE)
    std::map<std::string, ov::CompiledModel> compiled_models_;
};

} // namespace VPU
