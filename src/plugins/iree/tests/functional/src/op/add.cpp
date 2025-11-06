#include "operation_checker.hpp"

#include "openvino/runtime/infer_request.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(iree_plugin_tests, op_add_tensor_tensor) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    auto input1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, std::vector<float>{1.,2.,3.});
    auto add = std::make_shared<ov::op::v1::Add>(input0, input1);
    auto result0 = std::make_shared<ov::op::v0::Result>(add);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{3}, {-1.,0.,1.});
    test_suite.run();
}
