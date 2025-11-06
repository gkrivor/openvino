#include "operation_checker.hpp"

#include "openvino/runtime/infer_request.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(iree_plugin_tests, op_shapeof_v0) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3,2,1});
    auto shapeof = std::make_shared<ov::op::v0::ShapeOf>(input0);
    auto result0 = std::make_shared<ov::op::v0::Result>(shapeof);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{3,2,1}, {-1.,0.,1.,0.,1.,2.});
    test_suite.run();
}

TEST(iree_plugin_tests, op_shapeof_v3_i64) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3,2,1});
    auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(input0);
    auto result0 = std::make_shared<ov::op::v0::Result>(shapeof);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{3,2,1}, {-1.,0.,1.,0.,1.,2.});
    test_suite.run();
}

TEST(iree_plugin_tests, op_shapeof_v3_i32) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3,2,1});
    auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(input0, ov::element::i32);
    auto result0 = std::make_shared<ov::op::v0::Result>(shapeof);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{3,2,1}, {-1.,0.,1.,0.,1.,2.});
    test_suite.run();
}
