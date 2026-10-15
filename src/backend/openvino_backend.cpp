// VPU-CR-PROVENANCE
// Mechanism:  P3 staged local store (OpenVINO backend context)
// Derived-from: OpenVINO C++ API documentation (vendor specification, Class B)
// Spec entry: spec/backend.md#openvino
// Author:     agent:claude@opencodo-vpu-session-001
// Exposure:   none

#include "openvino_backend.hpp"

namespace VPU {

OpenVINOBackend::OpenVINOBackend(std::shared_ptr<Context> context,
                                  const std::string& device_name,
                                  ov::CompiledModel compiled_model,
                                  DeviceInfo device_info)
    : context_(std::move(context)),
      device_name_(device_name),
      compiled_model_(std::move(compiled_model)),
      device_info_(std::move(device_info)) {}

std::unique_ptr<OpenVINOBackend> OpenVINOBackend::Create(
    std::shared_ptr<Context> context,
    const std::string& device_name) {
    
    auto core = context->GetCore();
    
    // Query device capabilities
    auto device_info = context->GetDeviceInfo(device_name);
    
    // Compile model with default settings
    // In actual implementation, model path would be provided
    ov::CompiledModel empty_model;
    
    return std::unique_ptr<OpenVINOBackend>(new OpenVINOBackend(
        std::move(context), device_name, std::move(empty_model), std::move(device_info)));
}

void OpenVINOBackend::CompileModel(const std::string& model_path) {
    auto core = context_->GetCore();
    compiled_model_ = core->compile_model(model_path, device_name_);
}

std::unique_ptr<Tensor> OpenVINOBackend::RunInference(
    const std::string& model_path,
    const std::vector<std::unique_ptr<Tensor>>& inputs) {
    
    auto core = context_->GetCore();
    auto compiled = core->compile_model(model_path, device_name_);
    auto infer_request = compiled.create_infer_request();
    
    // Set inputs
    auto inputs_info = compiled.inputs();
    for (size_t i = 0; i < inputs.size() && i < inputs_info.size(); ++i) {
        if (inputs[i]) {
            auto ov_tensor = inputs[i]->ToOVTensor();
            infer_request.set_tensor(inputs_info[i].get_any_name(), ov_tensor);
        }
    }
    
    // Run inference
    infer_request.infer();
    
    // Get outputs
    auto outputs_info = compiled.outputs();
    if (outputs_info.empty()) {
        return nullptr;
    }
    
    auto output_tensor = infer_request.get_tensor(outputs_info[0].get_any_name());
    TensorSpec spec;
    spec.dtype = DType::FP32;
    spec.shape = output_tensor.get_shape();
    
    return Tensor::Create(spec, output_tensor.data());
}

} // namespace VPU