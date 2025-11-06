# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

set(TARGET_NAME ov_iree_func_tests)

ov_try_use_gold_linker()

ov_deprecated_no_errors()

message(STATUS "IREE Plugin test enabled")

if (ENABLE_STRICT_DEPENDENCIES)
    # For convinience add a runtime dependency to build along with this target.
    # Warning: Parallel build with -GNinja may not be efficient.
    list(APPEND ONNX_TESTS_DEPENDENCIES openvino_iree_plugin)
endif()

file (GLOB SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/*.cpp
)

list(APPEND SRC
    src/main.cpp
    src/operation_checker.cpp
)

# Create target

add_executable(${TARGET_NAME} ${SRC})
#add_test(NAME ov_onnx_frontend_tests COMMAND ov_onnx_frontend_tests --gtest_filter=-*IE_GPU*)
#set_property(TEST ov_onnx_frontend_tests PROPERTY LABELS OV UNIT ONNX_FE)

#add_dependencies(ov_onnx_frontend_tests openvino_template_extension)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

#target_compile_definitions(ov_onnx_frontend_tests
#    PRIVATE
#        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
#        SHARED_LIB_SUFFIX="${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

#set(ONNX_OPSET_VERSION 17 CACHE INTERNAL "Supported version of ONNX operator set")
#target_compile_definitions(ov_onnx_frontend_tests PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

#if(ONNX_TESTS_DEPENDENCIES)
#    add_dependencies(ov_onnx_frontend_tests ${ONNX_TESTS_DEPENDENCIES})
#endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    # gtest_main_manifest
    frontend_shared_test_classes
    func_test_utils)

if(OV_COMPILER_IS_CLANG)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

ov_build_target_faster(${TARGET_NAME} PCH)

# Install rules

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        DESTINATION tests/onnx
#        COMPONENT tests EXCLUDE_FROM_ALL
#        FILES_MATCHING
#        PATTERN "*.py"
#        PATTERN "*.onnx"
#        PATTERN "*.data")

#add_custom_command(TARGET ov_onnx_frontend_tests POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/unit_test.manifest"
#                                                    "${TEST_MODEL_ZOO_OUTPUT_DIR}/onnx/unit_test.manifest"
#                   ${custom_commands}
#                   COMMENT "Copy test manifest files to ${TEST_MODEL_ZOO}/onnx")

# Process models
#add_dependencies(ov_onnx_frontend_tests test_model_zoo)

# Working with ModelProto
#ov_link_system_libraries(ov_onnx_frontend_tests PUBLIC onnx_proto onnx)

#add_subdirectory(standalone_build)
#add_dependencies(ov_onnx_frontend_tests onnx_fe_standalone_build_test)
