# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(add_common_utils ADD_TARGET_NAME)
    list(APPEND TARGET_EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/gtest
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    if(NOT ENABLE_CONFORMANCE_PGQL)
        list(APPEND TARGET_EXCLUDED_SOURCE_PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/postgres_link.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/postgres_link.cpp
        )
    endif()

    # create target
    addIeTarget(
            NAME ${ADD_TARGET_NAME}
            TYPE STATIC
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            EXCLUDED_SOURCE_PATHS
                ${TARGET_EXCLUDED_SOURCE_PATHS}
            ADD_CPPLINT
            DEVELOPER_PACKAGE
                tests
            LINK_LIBRARIES
                PUBLIC
                    gtest
                    gtest_main
                    ngraphFunctions
                    openvino::runtime
                    openvino::runtime::dev
                PRIVATE
                    openvino::util
    )

    # USE_STATIC_IE is passed
    if(ARGN)
        target_link_libraries(${ADD_TARGET_NAME} PRIVATE inference_engine_s)
    else()
        target_link_libraries(${ADD_TARGET_NAME} PRIVATE inference_engine_legacy)
    endif()

    ie_faster_build(${ADD_TARGET_NAME}
        UNITY
        PCH PRIVATE "precomp.hpp"
    )

    # detecting regex support
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        set(USE_BOOST_RE ON)
    endif()
    if (USE_BOOST_RE)
        target_compile_definitions(${ADD_TARGET_NAME} PUBLIC USE_BOOST_RE)

        debug_message(STATUS "Adding boost dependency")
        if (CMAKE_VERBOSE_MAKEFILE)
            set(Boost_DEBUG on)
        endif ()
        find_package(Boost REQUIRED COMPONENTS regex)
        target_link_libraries(${ADD_TARGET_NAME} PUBLIC ${Boost_REGEX_LIBRARY})
        target_include_directories(${ADD_TARGET_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
    endif ()

    target_include_directories(${ADD_TARGET_NAME}
        PUBLIC
            $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE
            $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(${ADD_TARGET_NAME} SYSTEM PUBLIC ${IE_TESTS_ROOT}/ie_test_utils)

    target_compile_definitions(${ADD_TARGET_NAME} PUBLIC ${ARGN})

    if(ENABLE_CONFORMANCE_PGQL)
        target_compile_definitions(${ADD_TARGET_NAME} PRIVATE ENABLE_CONFORMANCE_PGQL)
        target_include_directories(${ADD_TARGET_NAME} PRIVATE
            ${POSTGRES_LIBPQ_INCLUDE}
            ${POSTGRES_LIBPQ_INCLUDE}/../../include
            $<TARGET_PROPERTY:openvino::pugixml,INTERFACE_INCLUDE_DIRECTORIES>
        )
        target_link_directories(${ADD_TARGET_NAME} PUBLIC ${POSTGRES_LIBPQ})
        if(WIN32)
            target_link_libraries(${ADD_TARGET_NAME} PUBLIC
                libpq.lib
                openvino::pugixml
            )
        else()
            target_link_libraries(${ADD_TARGET_NAME} PUBLIC
                pq
                openvino::pugixml
            )
        endif()
    endif()
endfunction()

set(TARGET_NAME commonTestUtils)

add_common_utils(${TARGET_NAME})
add_common_utils(${TARGET_NAME}_s USE_STATIC_IE)

add_subdirectory(tests)
