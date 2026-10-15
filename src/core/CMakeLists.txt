add_library(vpu_core STATIC
    context.cpp
)
target_include_directories(vpu_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENVINO_INCLUDE_DIRS}
)
target_link_libraries(vpu_core PUBLIC openvino::runtime)