#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "operation_checker.hpp"

TEST(iree_plugin_tests, op_reduce_mean_tensor_axes) {
    auto test_suite = ov::iree::OperationChecker();

    // data input (same as base example)
    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    // axes input: reduce along axis 0 of a 1-D tensor; integer element type per spec
    auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

    // keep_dims can be true or false; using true to keep output shape as {1} for a 1-D input
    auto reduce = std::make_shared<ov::op::v1::ReduceMean>(input0, axes, true);
    auto result0 = std::make_shared<ov::op::v0::Result>(reduce);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{3}, {-1.f, 0.f, 1.f});
    test_suite.run();
}
