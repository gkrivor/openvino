// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class LoRASubgraphHorizontalFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("LoRASubgraphHorizontalFusion");
    LoRASubgraphHorizontalFusion();
};

}  // namespace ov::intel_gpu
