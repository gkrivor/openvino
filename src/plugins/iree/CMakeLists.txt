if(NOT ENABLE_IREE)
    return()
endif()

set(TARGET_NAME "openvino_iree_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "IREE"
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

if(ENABLE_IREE_SRC)
    # iree compiler
    target_include_directories(${TARGET_NAME} PRIVATE "${ENABLE_IREE_SRC}/compiler/bindings/c")
    # iree runtime
    target_compile_definitions(${TARGET_NAME} PRIVATE IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl)
    target_include_directories(${TARGET_NAME} PRIVATE "${ENABLE_IREE_SRC}/runtime/src")
endif()

if(ENABLE_IREE_LIB)
    # iree compiler
    target_link_libraries(${TARGET_NAME} PRIVATE "${ENABLE_IREE_LIB}/lib/iree_compiler_bindings_c_loader.lib")
    # iree runtime
    target_link_libraries(${TARGET_NAME} PRIVATE "${ENABLE_IREE_LIB}/build_tools/third_party/flatcc/flatcc_parsing.lib")
    target_link_libraries(${TARGET_NAME} PRIVATE "${ENABLE_IREE_LIB}/runtime/src/iree/hal/local/elf/iree_hal_local_elf_arch.lib")
    target_link_libraries(${TARGET_NAME} PRIVATE "${ENABLE_IREE_LIB}/runtime/src/iree/runtime/iree_runtime_unified.lib")
endif()
