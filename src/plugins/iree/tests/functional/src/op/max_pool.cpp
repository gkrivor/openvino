#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "operation_checker.hpp"

TEST(iree_plugin_tests, op_maxpool_tensor) {
    auto test_suite = ov::iree::OperationChecker();

    // MaxPool takes a single data input; attributes define kernel/strides/pads.
    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 4, 4});

    ov::Strides strides{2, 2};
    ov::Shape pads_begin{0, 0};
    ov::Shape pads_end{0, 0};
    ov::Shape kernel{2, 2};

    auto maxpool = std::make_shared<ov::op::v1::MaxPool>(input0,
                                                         strides,
                                                         pads_begin,
                                                         pads_end,
                                                         kernel,
                                                         ov::op::RoundingType::FLOOR,
                                                         ov::op::PadType::EXPLICIT);

    auto result0 = std::make_shared<ov::op::v0::Result>(maxpool);
    test_suite.set_model(std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{input0}));

    // Provide a 1x1x4x4 input tensor.
    test_suite.add_input<float>(ov::Shape{1, 1, 4, 4},
                                {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});

    test_suite.run();
}
