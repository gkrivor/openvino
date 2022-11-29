# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#CMake Arguments:
# ENABLE_TESTS=ON - [REQUIRED] - Usage of postgres could be enabled only if ENABLE_TESTS is set to ON
# ENABLE_CONFORMANCE_PGQL=ON - [REQUIRED] - enables usage PostgresQL storage for conformance results
# POSTGRES_SRC=[path_to_postgres_sources] - [OPTIONAL] - default value is [openvino]/thirdparts/postgres/postgres
# POSTGRES_LIBPQ=[path_to_libpq.lib] - [OPTIONAL] - points on folder which contains libpq.lib
# POSTGRES_LIBPQ_INCLURE=[path_to_libpq-fe.h] - [OPTIONAL] - points on folder which contains libpq-fe.h

#On Linux you might be needed to install modules:
# apt-get install bison flex libreadline-dev

macro(libpq_build_windows)
    set(POSTGRES_LIBPQ "${POSTGRES_SRC}/${CMAKE_BUILD_TYPE}/libpq")
    set(POSTGRES_LIBPQ_INCLUDE "${POSTGRES_SRC}/src/interfaces/libpq")

    if(NOT EXISTS "${POSTGRES_LIBPQ}/libpq.lib" OR NOT EXISTS "${POSTGRES_LIBPQ}/libpq.dll")
        message(STATUS "Trying to build libpq...")
        message(STATUS "Looking for vcvarsall.bat...")
        # Looking for vcvarsall.bat file
        get_filename_component(MSVC_DIR
            ${CMAKE_CXX_COMPILER} DIRECTORY)
        string(FIND "${MSVC_DIR}" "/VC/" MSVC_VC_POS)
        if(MSVC_VC_POS EQUAL -1)
            message(FATAL_ERROR "Folder of VC isn't found in path ${MSVC_DIR}")
        endif()
        string(SUBSTRING "${MSVC_DIR}" 0 ${MSVC_VC_POS} MSVC_DIR)
        find_file(VCVARSALL_BAT
            NAMES vcvarsall.bat
            PATHS "${MSVC_DIR}/VC/Auxiliary/Build"
            NO_CACHE
            NO_DEFAULT_PATH)
        message(STATUS "Found at ${VCVARSALL_BAT}")

        # Setting up Visual Studio environment
        message(STATUS "Setting up Visual Studio environment...")
        execute_process(
            COMMAND cmd /C ${VCVARSALL_BAT} x86_amd64
            OUTPUT_VARIABLE _POSTGRES_VALUES
            ERROR_VARIABLE _POSTGRES_ERROR_VALUE
        )

        # Building libpq.dll
        message(STATUS "Building libpq...")
        execute_process(
            COMMAND cmd /C build.bat ${CMAKE_BUILD_TYPE} libpq
            WORKING_DIRECTORY "${POSTGRES_SRC}/src/tools/msvc/"
            RESULT_VARIABLE _LIBPQ_RESULT
            OUTPUT_VARIABLE _POSTGRES_VALUES
            ERROR_VARIABLE _POSTGRES_ERROR_VALUE
        )

        if(NOT _LIBPQ_RESULT MATCHES 0)
            message(FATAL_ERROR "PostgresQL: libpq hasn't be built, result: ${_LIBPQ_RESULT}")
        else()
            message(STATUS "PostgresQL: libpq has been built from sources")
        endif()
    endif() #NOT EXISTS libpq.lib OR libpq.dll
endmacro()

macro(libpq_build_linux)
    set(POSTGRES_LIBPQ "${POSTGRES_SRC}/src/interfaces/libpq")
    set(POSTGRES_LIBPQ_INCLUDE "${POSTGRES_SRC}/src/interfaces/libpq")

    if(NOT EXISTS "${POSTGRES_LIBPQ}/libpq.so")
        message(STATUS "Trying to build libpq...")

        # Setting up a build environment
        message(STATUS "Configuring a build environment...")
        execute_process(
            COMMAND bash ./configure
            WORKING_DIRECTORY "${POSTGRES_SRC}"
            OUTPUT_VARIABLE _POSTGRES_VALUES
            ERROR_VARIABLE _POSTGRES_ERROR_VALUE
        )

        # Building libpq.so
        message(STATUS "Building libpq...")
        execute_process(
            COMMAND make -C src/interfaces
            WORKING_DIRECTORY "${POSTGRES_SRC}"
            RESULT_VARIABLE _LIBPQ_RESULT
            OUTPUT_VARIABLE _POSTGRES_VALUES
            ERROR_VARIABLE _POSTGRES_ERROR_VALUE
        )

        if(NOT _LIBPQ_RESULT MATCHES 0)
            message(FATAL_ERROR "PostgresQL: libpq hasn't be built, result: ${_LIBPQ_RESULT}")
        else()
            message(STATUS "PostgresQL: libpq has been built from sources")
        endif()
    endif() #NOT EXISTS libpq.so
endmacro()

ie_option(ENABLE_CONFORMANCE_PGQL "Enables support of PostgreSQL-based reporting from test tools" OFF)

if(ENABLE_TESTS)
    message(STATUS "")
    message(STATUS "Additional test features:")
    if(ENABLE_CONFORMANCE_PGQL)
        message(STATUS "PostgresQL support .................... Enabled")
        if(NOT POSTGRES_LIBPQ OR NOT POSTGRES_LIBQP_INCLUDE)
            find_package(PostgreSQL)
            if(NOT PostgresQL_FOUND)
                message(STATUS "Trying to build PostgreSQL from source")
                if(NOT POSTGRES_SRC)
                    set(POSTGRES_SRC "${OpenVINO_SOURCE_DIR}/thirdparty/postgres/postgres")
                endif()
                if(WIN32)
                    libpq_build_windows()
                else()
                    libpq_build_linux()
                endif()
            else() #NOT PostgreSQL_FOUND
                set(POSTGRES_LIBPQ "${PostgreSQL_LIBRARIES}")
                set(POSTGRES_LIBPQ_INCLUDE "${PostgreSQL_INCLUDE_DIRS}")
            endif() #NOT PostgreSQL_FOUND
        endif() #NOT POSTGRES_LIBPQ_PATH OR NOT POSTGRES_LIBQP_INCLUDE

        if(POSTGRES_LIBPQ AND NOT POSTGRES_LIBPQ_INCLUDE)
            set(POSTGRES_LIBPQ_INCLUDE "${POSTGRES_LIBPQ}/../../src/interface/libpq")
        endif()
        
        if(NOT EXISTS "${POSTGRES_LIBPQ_INCLUDE}/libpq-fe.h")
            message(FATAL_ERROR "PostgresQL: libpq.h isn't found")
        endif()
        if(WIN32)
            if(NOT EXISTS "${POSTGRES_LIBPQ}/libpq.lib")
                message(FATAL_ERROR "PostgresQL: libpq.lib isn't found after a build")
            endif()
            if(NOT EXISTS "${POSTGRES_LIBPQ}/libpq.dll")
                message(FATAL_ERROR "PostgresQL: libpq.dll isn't found after a build")
            endif()
        else()
            if(NOT EXISTS "${POSTGRES_LIBPQ}/libpq.so")
                message(FATAL_ERROR "PostgresQL: libpq.so isn't found after a build")
            endif()
        endif()
        
        message(STATUS "POSTGRES_LIBPQ ........................ ${POSTGRES_LIBPQ}")
        message(STATUS "POSTGRES_LIBPQ_INCLUDE ................ ${POSTGRES_LIBPQ_INCLUDE}")
    endif() #ENABLE_CONFOFMANCE_PGQL
    message(STATUS "")
endif() #ENABLE_TESTS