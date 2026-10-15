// VPU-CR-PROVENANCE
// Mechanism:  R3 contract before timing (machine-readable workload contract)
// Derived-from: VPU_CLEANROOM_POLICY_v0.1.md section 5 (R3), VPU_CELL_CONCEPTS_v0.1 section 5 (manifest)
// Spec entry: spec/workload_contract.md#schema
// Author:     agent:claude@opencodo-vpu-session-001
// Exposure:   none

#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
namespace VPU {

// Forward declarations removed to avoid redefinition with tensor.hpp and context.hpp
struct WorkloadShape {
    size_t M = 1;
    size_t N = 512;
    size_t K = 2048;
    size_t seq_len = 1;
    size_t num_heads = 8;
    size_t head_dim = 64;
};

struct WorkloadDTypes {
    std::string compute = "fp32";
    std::string weight = "fp32";
    std::string activation = "fp32";
};

struct WorkloadTensorSpec {
    std::string name;
    size_t bytes = 0;
    std::vector<size_t> shape;
    std::string dtype;
};

struct WorkloadIO {
    std::vector<WorkloadTensorSpec> inputs;
    std::vector<WorkloadTensorSpec> outputs;
};

struct WorkloadSource {
    std::string paper_citation;
    std::string spec_entry;
};

struct WorkloadHash {
    std::string spirv = "";
    std::string source = "";
    std::string model = "";
};

struct WorkloadDeviceInfo {
    std::string gpu = "Intel Arc iGPU (128 EU)";
    std::string cpu = "Intel Core Ultra 5 135H";
    size_t memory_mb = 16384;
};

struct WorkloadTolerance {
    float max_error = 1e-4f;
    float rms_error = 1e-5f;
};

struct WorkloadContract {
    std::string contract_version = "1.0";
    std::string op;
    std::string description;
    WorkloadShape shape;
    WorkloadDTypes dtypes;
    WorkloadIO io;
    WorkloadSource source;
    WorkloadHash hash;
    WorkloadDeviceInfo device;
    WorkloadTolerance tolerance;
    std::vector<size_t> workgroup_shape = {8, 8, 1};
    size_t dispatch_count = 1;
    
    // Validation
    bool validate() const {
        if (op.empty()) throw std::runtime_error("Contract missing 'op'");
        if (io.inputs.empty()) throw std::runtime_error("Contract missing input specification");
        if (io.outputs.empty()) throw std::runtime_error("Contract missing output specification");
        if (hash.source.empty()) throw std::runtime_error("Contract missing source hash");
        return true;
    }
};

} // namespace VPU