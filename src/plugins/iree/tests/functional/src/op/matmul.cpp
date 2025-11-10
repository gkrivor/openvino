#include "openvino/op/matmul.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "operation_checker.hpp"

TEST(iree_plugin_tests, op_matmul_tensor_tensor) {
    auto test_suite = ov::iree::OperationChecker();

    // A: [2, 3], B: [3, 4] -> Y: [2, 4]
    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto input1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                         ov::Shape{3, 4},
                                                         std::vector<float>{// 3x4 matrix in row-major order
                                                                            1.f,
                                                                            2.f,
                                                                            3.f,
                                                                            4.f,
                                                                            5.f,
                                                                            6.f,
                                                                            7.f,
                                                                            8.f,
                                                                            9.f,
                                                                            10.f,
                                                                            11.f,
                                                                            12.f});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input0, input1);
    auto result0 = std::make_shared<ov::op::v0::Result>(matmul);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    test_suite.add_input<float>(ov::Shape{2, 3},
                                {// 2x3 matrix in row-major order
                                 -1.f,
                                 0.f,
                                 1.f,
                                 2.f,
                                 -2.f,
                                 3.f});
    test_suite.run();
}
