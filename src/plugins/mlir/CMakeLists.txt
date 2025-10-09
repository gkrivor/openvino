if(NOT ENABLE_MLIR)
    return()
endif()

set(TARGET_NAME "openvino_mlir_plugin")

add_library(${TARGET_NAME} SHARED
    src/plugin.cpp
    src/compiled_model.cpp
    src/infer_request.cpp
    src/ops/abs.cpp
    src/ops/add.cpp
    src/ops/mul.cpp
    src/ops/parameter.cpp
    src/ops/result.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev)
target_include_directories(${TARGET_NAME} PRIVATE ${OpenVINO_INCLUDE_DIRS})
target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)


ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
