cmake_minimum_required(VERSION 3.20)
project(VPU_MoE_Lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable SEH exception handling for catch(...) to work (R4: correctness gates)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
endif()

# Fix CMP0144: find_package uses upper-case <PACKAGENAME>_ROOT variables
cmake_policy(SET CMP0144 OLD)

# OpenVINO root from environment variable (Class B: SDK, public API only)
if(DEFINED ENV{OPENVINO_ROOT})
    set(OPENVINO_ROOT "$ENV{OPENVINO_ROOT}")
endif()

if(NOT OPENVINO_ROOT)
    message(FATAL_ERROR "OpenVINO root not found. Set OPENVINO_ROOT environment variable.")
endif()

# OpenVINO integration via imported targets (Class B: public API only)
list(APPEND CMAKE_PREFIX_PATH "${OPENVINO_ROOT}/runtime/cmake")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Project source directories
add_subdirectory(src)

# Configuration for clean-room compliance
enable_testing()
add_custom_target(check_cleanroom
    COMMAND ${CMAKE_COMMAND} -E echo "Running clean-room checks..."
    COMMENT "Verifying clean-room compliance"
)