add_library(vpu_backend STATIC
    openvino_backend.cpp
)
target_include_directories(vpu_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vpu_backend PUBLIC vpu_core vpu_model openvino::runtime)