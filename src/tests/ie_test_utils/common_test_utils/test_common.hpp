// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>
#include "test_assertions.hpp"

namespace CommonTestUtils {

class PostgreSQLHandler {
protected:
    PostgreSQLHandler();
    virtual ~PostgreSQLHandler();

    virtual void SetUp();
    virtual void TearDown();
};

class TestsCommon : virtual public ::testing::Test, public PostgreSQLHandler {
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;

    virtual void SetUp() override {
        PostgreSQLHandler::SetUp();
    }
    virtual void TearDown() override {
        PostgreSQLHandler::TearDown();
    }

    friend class PostgreSQLHandler;
};

}  // namespace CommonTestUtils
