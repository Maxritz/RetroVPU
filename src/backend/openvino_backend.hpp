// VPU-CR-PROVENANCE
// Mechanism:  P3 staged local store (OpenVINO backend context)
// Derived-from: OpenVINO C++ API documentation (vendor specification, Class B)
// Spec entry: spec/backend.md#openvino
// Author:     agent:claude@opencodo-vpu-session-001
// Exposure:   none

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "context.hpp"
#include "tensor.hpp"

namespace VPU {

class OpenVINOBackend {
public:
    static std::unique_ptr<OpenVINOBackend> Create(std::shared_ptr<Context> context,
                                                    const std::string& device_name);
    
    std::unique_ptr<Tensor> RunInference(const std::string& model_path, 
                                          const std::vector<std::unique_ptr<Tensor>>& inputs);
    
    void CompileModel(const std::string& model_path);
    
    const DeviceInfo& device_info() const { return device_info_; }
    ov::CompiledModel compiled_model() const { return compiled_model_; }
    
    // V12 GPU dispatch methods (stubs: return empty to trigger CPU fallback)
    std::vector<float> RunDot(size_t N, const float* x, const float* y, const std::string& device) {
        (void)N; (void)x; (void)y; (void)device; return {};
    }
    std::vector<float> RunReduceSum(size_t N, const float* data, const std::string& device) {
        (void)N; (void)data; (void)device; return {};
    }
    std::vector<float> RunAmax(size_t N, const float* x, const std::string& device) {
        (void)N; (void)x; (void)device; return {};
    }
    std::vector<float> RunScal(size_t N, float alpha, float* x, const std::string& device) {
        (void)N; (void)alpha; (void)x; (void)device; return {};
    }
    std::vector<float> RunAxpy(size_t N, float alpha, const float* x, float* y, const std::string& device) {
        (void)N; (void)alpha; (void)x; (void)y; (void)device; return {};
    }
    std::unique_ptr<Tensor> RunGemv(bool trans, size_t M, size_t N, float alpha, float beta,
                                    const float* A, const float* x, float* y, const std::string& device) {
        (void)trans; (void)M; (void)N; (void)alpha; (void)beta; (void)A; (void)x; (void)y; (void)device;
        return nullptr;
    }
    std::vector<double> RunGemvFP64(bool trans, size_t M, size_t N, double alpha, double beta,
                                    const double* A, const double* x, double* y, const std::string& device) {
        (void)trans; (void)M; (void)N; (void)alpha; (void)beta; (void)A; (void)x; (void)y; (void)device;
        return {};
    }
    std::vector<float> RunGer(size_t M, size_t N, float alpha, const float* x,
                              const float* y, float* A, const std::string& device) {
        (void)M; (void)N; (void)alpha; (void)x; (void)y; (void)A; (void)device; return {};
    }
    std::vector<float> RunBatchGemv(size_t batch_size, size_t M, size_t N, float alpha,
                                    const float* A_batch, const float* x_batch,
                                    float* y_batch, const std::string& device) {
        (void)batch_size; (void)M; (void)N; (void)alpha; (void)A_batch; (void)x_batch;
        (void)y_batch; (void)device; return {};
    }
    
private:
    OpenVINOBackend(std::shared_ptr<Context> context, 
                    const std::string& device_name,
                    ov::CompiledModel compiled_model,
                    DeviceInfo device_info);
    
    std::shared_ptr<Context> context_;
    std::string device_name_;
    ov::CompiledModel compiled_model_;
    DeviceInfo device_info_;
};

} // namespace VPU