// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>

#include "test_assertions.hpp"

namespace CommonTestUtils {
#ifdef ENABLE_CONFORMANCE_PGQL
class PostgreSQLLink;
#endif

class TestsCommon : virtual public ::testing::Test {
#ifdef ENABLE_CONFORMANCE_PGQL
    PostgreSQLLink* PGLink;
#endif
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;

#ifdef ENABLE_CONFORMANCE_PGQL
public:
    PostgreSQLLink* GetPGLink() {
        assert(this->PGLink != nullptr);
        return this->PGLink;
    }
#endif
};

}  // namespace CommonTestUtils
