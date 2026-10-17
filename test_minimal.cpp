#include <iostream>
#include <openvino/openvino.hpp>

int main() {
    std::cout << "Step 1: Starting" << std::endl;
    std::cout.flush();
    
    try {
        std::cout << "Step 2: Creating Core" << std::endl;
        std::cout.flush();
        
        ov::Core core;
        
        std::cout << "Step 3: Core created" << std::endl;
        std::cout.flush();
        
        auto devices = core.get_available_devices();
        std::cout << "Step 4: Devices = " << devices.size() << std::endl;
        for (auto& d : devices) std::cout << "  " << d << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
