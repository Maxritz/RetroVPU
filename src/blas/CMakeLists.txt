# VPU-CR-PROVENANCE: author=deploy3-integration
# Clean-room origin: this project (V12-VPU clean-room).
# CMake build for the V12-VPU BLAS library (Levels 1, 2, 3 + memory dispatch).
# OpenBLAS is the independent standard counterpart baseline (R5, R6).

cmake_minimum_required(VERSION 3.21)

# OpenBLAS → GPU Porting with V12 Methods
# Clean-room implementation of BLAS algorithms adapted to GPU with V12 optimizations
add_library(vpu_blas STATIC
    vpu_level1.cpp
    vpu_level2.cpp
    vpu_level3.cpp
    vpu_memory.cpp
    vpu_dispatcher.cpp
    optimized_baseline.cpp
    hgemv_tile_1_enhanced.cpp
)

# Local blas/ headers + project src/ headers
# (enables "harness/benchmark.hpp", "core/context.hpp", "model/tensor.hpp" includes)
target_include_directories(vpu_blas PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(vpu_blas PUBLIC cxx_std_17)

# Enable AVX2+AVX-VNNI for optimized baseline (R10: cost-gated, CPU-only paths)
# Intel Core Ultra 5 135H supports AVX2 + FMA3 + AVX-VNNI but NOT AVX-512
if(MSVC)
    target_compile_options(vpu_blas PRIVATE /arch:AVX2)
else()
    target_compile_options(vpu_blas PRIVATE -mavx2 -mfma)
endif()

# Link against existing project libraries (Class B: OpenVINO SDK headers only)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(vpu_blas PUBLIC
    vpu_core
    vpu_model
    vpu_backend
    vpu_harness
    openvino::runtime
)

# OpenBLAS is the independent standard counterpart baseline (R5, R6).
# Link is optional — the dispatcher estimates baseline timing when
# OpenBLAS is unavailable and falls back to the cost model.
find_package(BLAS QUIET)
if(BLAS_FOUND)
    target_link_libraries(vpu_blas PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(vpu_blas PRIVATE VPU_HAVE_OPENBLAS)
endif()
