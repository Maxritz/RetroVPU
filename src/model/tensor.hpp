// VPU-CR-PROVENANCE
// Mechanism:  P3 staged local store (tensor layout for efficient memory staging)
// Derived-from: arXiv 0910.2324, section 4 (data movement and layout)
// Spec entry: spec/tensor.md#layout
// Author:     agent:claude@opencodo-vpu-session-001
// Exposure:   none

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <openvino/openvino.hpp>

namespace VPU {

enum class DType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4
};

struct TensorSpec {
    DType dtype;
    std::vector<size_t> shape;
    size_t num_bytes() const;
    std::string to_string() const;
};

class Tensor {
public:
    ~Tensor();
    static std::unique_ptr<Tensor> Create(const TensorSpec& spec, void* data = nullptr, size_t owner_size = 0);
    
    std::unique_ptr<Tensor> Clone() const;
    ov::Tensor ToOVTensor() const;
    const TensorSpec& spec() const { return spec_; }
    void* data() const { return data_; }
    
private:
    Tensor(TensorSpec spec, void* data, size_t owner_size);
    TensorSpec spec_;
    void* data_;
    size_t owner_size_;
};

} // namespace VPU