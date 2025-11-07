#include "openvino/op/convolution.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "operation_checker.hpp"

// 1D Convolution: data (N,C,W), filters (O,I,K)
TEST(iree_plugin_tests, op_convolution1d_tensor_tensor) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 7});  // N=1, C=2, W=7
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 3});  // O=3, I=2, K=3

    auto conv = std::make_shared<ov::op::v1::Convolution>(input0,
                                                          input1,
                                                          ov::Strides{1},
                                                          ov::CoordinateDiff{1},
                                                          ov::CoordinateDiff{1},
                                                          ov::Strides{1},
                                                          ov::op::PadType::EXPLICIT);

    auto result0 = std::make_shared<ov::op::v0::Result>(conv);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0, input1}));

    test_suite.add_input<float>(ov::Shape{1, 2, 7}, std::vector<float>(1 * 2 * 7, 1.0f));
    test_suite.add_input<float>(ov::Shape{3, 2, 3}, std::vector<float>(3 * 2 * 3, 0.5f));
    test_suite.run();
}

// 2D Convolution: data (N,C,H,W), filters (O,I,KY,KX)
TEST(iree_plugin_tests, op_convolution2d_tensor_tensor) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5, 5});  // N=1, C=3, H=W=5
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 3, 3});  // O=2, I=3, 3x3

    auto conv = std::make_shared<ov::op::v1::Convolution>(input0,
                                                          input1,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1},
                                                          ov::op::PadType::EXPLICIT);

    auto result0 = std::make_shared<ov::op::v0::Result>(conv);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0, input1}));

    test_suite.add_input<float>(ov::Shape{1, 3, 5, 5}, std::vector<float>(1 * 3 * 5 * 5, 1.0f));
    test_suite.add_input<float>(ov::Shape{2, 3, 3, 3}, std::vector<float>(2 * 3 * 3 * 3, 0.5f));
    test_suite.run();
}

// 3D Convolution: data (N,C,D,H,W), filters (O,I,KD,KH,KW)
TEST(iree_plugin_tests, op_convolution3d_tensor_tensor) {
    auto test_suite = ov::iree::OperationChecker();

    auto input0 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 6, 6, 6});  // N=1, C=2, D=H=W=6
    auto input1 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 2, 3, 3, 3});  // O=4, I=2, 3x3x3

    auto conv = std::make_shared<ov::op::v1::Convolution>(input0,
                                                          input1,
                                                          ov::Strides{1, 1, 1},
                                                          ov::CoordinateDiff{1, 1, 1},
                                                          ov::CoordinateDiff{1, 1, 1},
                                                          ov::Strides{1, 1, 1},
                                                          ov::op::PadType::EXPLICIT);

    auto result0 = std::make_shared<ov::op::v0::Result>(conv);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0, input1}));

    test_suite.add_input<float>(ov::Shape{1, 2, 6, 6, 6}, std::vector<float>(1 * 2 * 6 * 6 * 6, 1.0f));
    test_suite.add_input<float>(ov::Shape{4, 2, 3, 3, 3}, std::vector<float>(4 * 2 * 3 * 3 * 3, 0.5f));
    test_suite.run();
}
