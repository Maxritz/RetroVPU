// VPU-CR-PROVENANCE
// Mechanism:  P3 staged local store (adapted for OpenVINO context management)
// Derived-from: arXiv 0910.2324, section 3 (local store management)
// Spec entry: spec/context.md#device-query
// Author:     agent:claude@opencodo-vpu-session-001
// Exposure:   none

#include "context.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace VPU {

Context::Context(std::shared_ptr<ov::Core> core) : core_(std::move(core)) {}

std::unique_ptr<Context> Context::Create() {
    auto core = std::make_shared<ov::Core>();
    return std::unique_ptr<Context>(new Context(core));
}

size_t Context::GetSystemMemoryMB() const {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<size_t>(memInfo.ullTotalPhys / (1024 * 1024));
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages * page_size / (1024 * 1024));
    }
    return 16384; // fallback
#endif
}

std::vector<DeviceInfo> Context::GetAvailableDevices() const {
    std::vector<DeviceInfo> devices;
    auto device_names = core_->get_available_devices();
    
    for (const auto& name : device_names) {
        DeviceInfo info;
        info.name = name;
        info.memory_mb = 0;
        info.compute_units = 0;
        
        // Determine device type from name
        if (name.find("CPU") != std::string::npos) {
            info.type = Device::CPU;
        } else if (name.find("NPU") != std::string::npos) {
            info.type = Device::NPU;
        } else if (name.find("GPU") != std::string::npos) {
            info.type = Device::GPU;
        }
        
        // Query device-specific memory and compute info
        try {
            // Query optimal number of threads
            try {
                int optimal = core_->get_property(name, "OPTIMAL_NUMBER_OF_THREADS").as<int>();
                info.compute_units = static_cast<size_t>(optimal);
            } catch (...) {
                info.compute_units = std::thread::hardware_concurrency();
            }
            
            // Query device memory
            if (info.type == Device::CPU) {
                info.memory_mb = GetSystemMemoryMB();
            } else if (info.type == Device::GPU) {
                // Arc iGPU shares system memory with CPU - do not double-count
                info.memory_mb = 0;  // shared pool, see CPU entry
            } else if (info.type == Device::NPU) {
                // NPU has access to 8 GB shared system memory
                info.memory_mb = 8192;
            }
            
        } catch (...) {
            // Fallback values from project specification
            if (info.type == Device::CPU) {
                info.memory_mb = 16384;  // 16 GB RAM
                info.compute_units = std::thread::hardware_concurrency();
            } else if (info.type == Device::GPU) {
                info.memory_mb = 0;  // shared with CPU
                info.compute_units = 128;
            } else if (info.type == Device::NPU) {
                info.memory_mb = 8192;  // NPU can address 8 GB
                info.compute_units = 1;
            }
        }
        
        devices.push_back(info);
    }
    
    // If no devices found, add known hardware from spec
    // If no devices found, add known hardware from spec
    if (devices.empty()) {
        DeviceInfo cpu_info;
        cpu_info.name = "CPU";
        cpu_info.type = Device::CPU;
        cpu_info.memory_mb = GetSystemMemoryMB();
        cpu_info.compute_units = std::thread::hardware_concurrency();
        devices.push_back(cpu_info);
    }
    
    return devices;
}

DeviceInfo Context::GetDeviceInfo(const std::string& device_name) const {
    auto devices = GetAvailableDevices();
    for (auto& dev : devices) {
        if (dev.name.find(device_name) != std::string::npos) {
            return dev;
        }
    }
    
    // Default fallback
    DeviceInfo info;
    info.name = device_name;
    info.memory_mb = 0;
    info.compute_units = 0;
    
    if (device_name.find("CPU") != std::string::npos) {
        info.type = Device::CPU;
        info.memory_mb = GetSystemMemoryMB();
        info.compute_units = std::thread::hardware_concurrency();
    } else if (device_name.find("NPU") != std::string::npos) {
        info.type = Device::NPU;
        info.memory_mb = 8192;
    } else if (device_name.find("GPU") != std::string::npos) {
        info.type = Device::GPU;
        info.memory_mb = 0;  // shared with CPU
        info.compute_units = 128;
    }
    
    return info;
}

} // namespace VPU
