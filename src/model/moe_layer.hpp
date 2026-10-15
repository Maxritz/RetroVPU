// VPU-CR-PROVENANCE
// Mechanism:  P10 sparse compute; P16 placement-aware dispatch order
// Derived-from: arXiv 0710.2442, section 5 (parallel task scheduling); VPU_CELL_CONCEPTS_v0.1, section 5 (H-E, bounded claim)
// Spec entry: spec/moe_layer.md#dispatch
// Author:     agent:claude@opencodo-vpu-session-001
// Exposure:   none

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <random>
#include "tensor.hpp"

namespace VPU {

struct MoELayerConfig {
    size_t num_experts;
    size_t expert_capacity;
    size_t hidden_size;
    size_t intermediate_size;
    DType compute_dtype;
    bool use_fp16_accumulation;
    size_t top_k; // number of experts per token
};

struct MoEDispatchResult {
    std::vector<size_t> expert_assignment;
    std::vector<float> routing_weights;
    std::vector<size_t> token_indices;
};

class MoELayer {
public:
    static std::unique_ptr<MoELayer> Create(const MoELayerConfig& config);
    
    MoEDispatchResult DispatchTokens(const Tensor& input);
    std::unique_ptr<Tensor> ComputeExpert(size_t expert_idx, const Tensor& tokens);
    std::unique_ptr<Tensor> MergeExperts(const std::vector<std::unique_ptr<Tensor>>& expert_outputs, 
                                          const MoEDispatchResult& dispatch_result,
                                          const Tensor& original_input);
    
    std::unique_ptr<Tensor> Forward(const Tensor& input);
    
    const MoELayerConfig& config() const { return config_; }
    
private:
    MoELayer(MoELayerConfig config);
    MoELayerConfig config_;
    std::vector<std::vector<uint8_t>> expert_weights_; // packed quantized weights
    std::mt19937 rng_;
};

} // namespace VPU