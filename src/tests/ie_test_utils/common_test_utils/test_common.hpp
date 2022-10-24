// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>
#include "test_assertions.hpp"
#ifdef ENABLE_CONFORMANCE_PGQL
#include <map>
#endif

namespace CommonTestUtils {
#ifdef ENABLE_CONFORMANCE_PGQL
struct PostgreSQLCustomData;
class PostgreSQLHandler : virtual public ::testing::Test {
    PostgreSQLCustomData* customData;

protected:
    PostgreSQLHandler();
    ~PostgreSQLHandler() override;

    bool SetCustomField(const std::string fieldName, const std::string fieldValue, const bool rewrite = true);
    std::string GetCustomField(const std::string fieldName, const std::string defaultValue = "") const;
    bool RemoveCustomField(const std::string fieldName);
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
