// VPU-CR-PROVENANCE
// Mechanism:  P3 staged local store (tensor layout for efficient memory staging)
// Derived-from: arXiv 0910.2324, section 4 (data movement and layout)
// Spec entry: spec/tensor.md#layout
// Author:     agent:claude@opencode-vpu-session-001
// Exposure:   none

#include "tensor.hpp"
#include <cstring>

namespace VPU {

size_t TensorSpec::num_bytes() const {
    size_t element_size = 4; // default FP32
    switch (dtype) {
        case DType::FP16: element_size = 2; break;
        case DType::BF16: element_size = 2; break;
        case DType::INT8: element_size = 1; break;
        case DType::INT4: element_size = 0; break; // packed
        default: element_size = 4; break;
    }
    
    if (dtype == DType::INT4) {
        // INT4 is packed 2:1
        return (shape.empty() ? 1 : shape[0]) / 2;
    }
    
    size_t total = element_size;
    for (auto dim : shape) {
        total *= dim;
    }
    return total;
}

std::string TensorSpec::to_string() const {
    std::string type_str;
    switch (dtype) {
        case DType::FP32: type_str = "f32"; break;
        case DType::FP16: type_str = "f16"; break;
        case DType::BF16: type_str = "bf16"; break;
        case DType::INT8: type_str = "i8"; break;
        case DType::INT4: type_str = "i4"; break;
    }
    return type_str + "[" + std::to_string(shape.size()) + "d]";
}

Tensor::Tensor(TensorSpec spec, void* data, size_t owner_size) 
    : spec_(std::move(spec)), data_(data), owner_size_(owner_size) {}

Tensor::~Tensor() {
    if (data_ && owner_size_ > 0) {
        delete[] static_cast<uint8_t*>(data_);
    }
}

std::unique_ptr<Tensor> Tensor::Create(const TensorSpec& spec, void* data, size_t owner_size) {
    return std::unique_ptr<Tensor>(new Tensor(spec, data, owner_size));
}

ov::Tensor Tensor::ToOVTensor() const {
    ov::element::Type type;
    switch (spec_.dtype) {
        case DType::FP32: type = ov::element::f32; break;
        case DType::FP16: type = ov::element::f16; break;
        case DType::BF16: type = ov::element::bf16; break;
        case DType::INT8: type = ov::element::i8; break;
        default: type = ov::element::f32; break;
    }
    
    ov::Shape shape(spec_.shape.begin(), spec_.shape.end());
    return ov::Tensor(type, shape, data_);
}

std::unique_ptr<Tensor> Tensor::Clone() const {
    if (owner_size_ > 0 && data_) {
        // Deep copy owned data
        uint8_t* new_data = new uint8_t[owner_size_];
        std::memcpy(new_data, data_, owner_size_);
        return Tensor::Create(spec_, new_data, owner_size_);
    }
    return Tensor::Create(spec_, data_);
}

} // namespace VPU