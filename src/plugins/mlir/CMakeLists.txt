cmake_minimum_required(VERSION 3.13)
project(mlir_plugin)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(openvino_mlir_plugin SHARED
    src/plugin.cpp
    src/compiled_model.cpp
    src/infer_request.cpp
    src/ops/add.cpp
    src/ops/mul.cpp
    src/ops/parameter.cpp
    src/ops/result.cpp
)

target_link_libraries(openvino_mlir_plugin PRIVATE openvino::runtime)
target_include_directories(openvino_mlir_plugin PRIVATE ${OpenVINO_INCLUDE_DIRS})

set_target_properties(openvino_mlir_plugin PROPERTIES
    OUTPUT_NAME "openvino_mlir_plugin"
)
