// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>
#include "test_assertions.hpp"

namespace CommonTestUtils {
#ifdef ENABLE_CONFORMANCE_PGQL
class PostgreSQLHandler : virtual public ::testing::Test {
protected:
    PostgreSQLHandler();
    virtual ~PostgreSQLHandler();
};
#endif

class TestsCommon :
#ifndef ENABLE_CONFORMANCE_PGQL
    virtual public ::testing::Test
#else
    virtual public PostgreSQLHandler
#endif
{
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
};

}  // namespace CommonTestUtils
