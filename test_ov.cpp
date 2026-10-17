#include <openvino/openvino.hpp>
#include <iostream>
int main() {
    std::cout << "Before Core" << std::endl;
    try {
        ov::Core core;
        std::cout << "Core created" << std::endl;
        auto devices = core.get_available_devices();
        std::cout << "Devices: ";
        for (auto& d : devices) std::cout << d << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
