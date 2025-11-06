#include "operation_checker.hpp"

#include "openvino/runtime/infer_request.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(iree_plugin_tests, op_abs) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    input0->output(0).add_names({"input0"});
    input0->set_friendly_name("input0");
    auto abs = std::make_shared<ov::op::v0::Abs>(input0);
    input0->output(0).add_names({"operation"});
    abs->set_friendly_name("operation");
    auto result0 = std::make_shared<ov::op::v0::Result>(abs);
    result0->output(0).add_names({"operation"});
    result0->set_friendly_name("output0");
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{3}, {-1.,0.,1.});
    test_suite.run();
}