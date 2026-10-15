add_library(vpu_model STATIC
    tensor.cpp
    moe_layer.cpp
)
target_include_directories(vpu_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vpu_model PUBLIC vpu_core openvino::runtime)