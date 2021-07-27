// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
namespace ov {
namespace test {
namespace subgraph {

using ReadIRParams = std::tuple<
        std::string,                         // IR path
        std::string,                         // Target Device
        ov::AnyMap>;                         // Plugin Config

class ReadIRBase : public ov::test::SubgraphBaseTest {
public:
    void GenerateInputs() override;

    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expected,
                 const std::vector<InferenceEngine::Blob::Ptr> &actual) override;
    std::vector<InferenceEngine::Blob::Ptr> GetOutputs() override;
};

class ReadIRTest : public testing::WithParamInterface<ReadIRParams>,
                   virtual public ReadIRBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj);
    void query_model() override;

protected:
    void SetUp() override;

private:
    std::string pathToModel;
    std::string sourceModel;
    std::vector<std::pair<std::string, size_t>> ocuranceInModels;
};
} // namespace subgraph
} // namespace test
} // namespace ov
