# VPU MoE Lite - Source build configuration
# OpenBLAS → GPU Porting with V12 Methods

add_subdirectory(core)
add_subdirectory(model)
add_subdirectory(backend)
add_subdirectory(harness)
add_subdirectory(blas)

add_executable(vpu_moe_lite
    main.cpp
)
target_link_libraries(vpu_moe_lite
    vpu_core
    vpu_model
    vpu_backend
    vpu_harness
    vpu_blas
    openvino::runtime
)

# R1-R12 compliance benchmark suite for OpenBLAS → GPU porting
add_executable(benchmark_vpu_blas
    benchmark_vpu_blas.cpp
)
target_link_libraries(benchmark_vpu_blas
    vpu_core
    vpu_model
    vpu_backend
    vpu_harness
    vpu_blas
    openvino::runtime
)