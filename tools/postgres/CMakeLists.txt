# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_custom_target(postgres)

add_custom_command(TARGET postgres
    POST_BUILD
    COMMAND cmd /c \"\"src/tools/msvc/build.bat\" ${CMAKE_BUILD_TYPE} libpq\"
    WORKING_DIRECTORY "${OpenVINO_SOURCE_DIR}/thirdparty/postgres/postgres"
    COMMENT "Building PostgresQL library"
)
