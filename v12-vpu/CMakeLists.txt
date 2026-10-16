cmake_minimum_required(VERSION 3.21)
project(v12-vpu-bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan SDK
find_package(Vulkan REQUIRED)

add_executable(vulkan_bench
    benchmarks/vulkan_bench.cpp
)
target_include_directories(vulkan_bench PRIVATE ${Vulkan_INCLUDE_DIR})
target_link_libraries(vulkan_bench PRIVATE ${Vulkan_LIBRARY})

# Compiler flags
if(MSVC)
    target_compile_options(vulkan_bench PRIVATE /O2 /EHsc /W3 /DUNICODE /D_UNICODE)
endif()
