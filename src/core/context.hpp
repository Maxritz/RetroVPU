// VPU-CR-PROVENANCE
// Mechanism:  P3 staged local store (adapted for OpenVINO context management)
// Derived-from: arXiv 0910.2324, section 3 (local store management)
// Spec entry: spec/context.md#device-query
// Author:     agent:claude@opencode-vpu-session-001
// Exposure:   none

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <openvino/openvino.hpp>

namespace VPU {

enum class Device {
    CPU,
    NPU,
    GPU
};

struct DeviceInfo {
    std::string name;
    Device type;
    size_t memory_mb;
    size_t compute_units;
};

class Context {
public:
    static std::unique_ptr<Context> Create();
    
    std::vector<DeviceInfo> GetAvailableDevices() const;
    DeviceInfo GetDeviceInfo(const std::string& device_name) const;
    size_t GetSystemMemoryMB() const;
    
    std::shared_ptr<ov::Core> GetCore() const { return core_; }
    
private:
    Context(std::shared_ptr<ov::Core> core);
    std::shared_ptr<ov::Core> core_;
};

} // namespace VPU