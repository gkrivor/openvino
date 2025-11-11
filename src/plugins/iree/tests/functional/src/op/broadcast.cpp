#include "openvino/op/broadcast.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "operation_checker.hpp"

TEST(iree_plugin_tests, op_broadcast_tensor_targetshape_numpy) {
    auto test_suite = ov::iree::OperationChecker();

    // data: f32[3]
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    // target_shape: i64[1] -> [3]
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Broadcast-3 in NUMPY mode (axes_mapping not needed)
    auto bcast = std::make_shared<ov::op::v3::Broadcast>(data, target_shape);
    auto result0 = std::make_shared<ov::op::v0::Result>(bcast);
    test_suite.set_model(
        std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{data, target_shape}));

    // Provide runtime inputs
    test_suite.add_input<float>(ov::Shape{3}, {-1.f, 0.f, 1.f});  // data
    test_suite.add_input<int64_t>(ov::Shape{1}, {3});             // target_shape = [3]
    test_suite.run();
}
