#include "openvino/op/squeeze.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "operation_checker.hpp"

TEST(iree_plugin_tests, op_squeeze_tensor) {
    auto test_suite = ov::iree::OperationChecker();

    // Input data with singleton dims to squeeze: shape [1, 3, 1]
    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 1});
    // Squeeze axes: remove dims 0 and 2 -> result shape [3]
    auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 2});
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(input0, axes);
    auto result0 = std::make_shared<ov::op::v0::Result>(squeeze);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{1, 3, 1}, {-1.f, 0.f, 1.f});
    test_suite.run();
}
