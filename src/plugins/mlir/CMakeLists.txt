if(NOT ENABLE_MLIR)
    return()
endif()

set(TARGET_NAME "openvino_mlir_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "MLIR"
              PSEUDO_DEVICE
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR src/plugin.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev)
target_include_directories(${TARGET_NAME} PRIVATE ${OpenVINO_INCLUDE_DIRS})
target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER=\"mock_version\")

