# VPU MoE Lite - Source build configuration

add_subdirectory(core)
add_subdirectory(model)
add_subdirectory(backend)
add_subdirectory(harness)

add_executable(vpu_moe_lite
    main.cpp
)
target_link_libraries(vpu_moe_lite
    vpu_core
    vpu_model
    vpu_backend
    vpu_harness
    openvino::runtime
)