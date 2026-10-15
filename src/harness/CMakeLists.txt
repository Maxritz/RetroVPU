# BenchmarkHarness is header-only (template-based benchmark)
add_library(vpu_harness INTERFACE)
target_include_directories(vpu_harness INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vpu_harness INTERFACE vpu_core vpu_model openvino::runtime)