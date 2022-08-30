# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME postgresTest)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
        INCLUDES
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADD_CPPLINT
        LABELS
            POSTGRES_TEST
)

ie_faster_build(${TARGET_NAME} UNITY)

if(ENABLE_CONFORMANCE_PGQL)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_CONFORMANCE_PGQL)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${POSTGRES_LIBPQ_INCLUDE}
        ${POSTGRES_LIBPQ_INCLUDE}/../../include
    )
    target_link_directories(${TARGET_NAME} PRIVATE ${POSTGRES_LIBPQ})
    if(WIN32)
        target_link_libraries(${TARGET_NAME} libpq.lib)
    endif()
endif()

