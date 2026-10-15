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