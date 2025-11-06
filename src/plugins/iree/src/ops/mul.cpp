#include "plugin/iree/common.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov::iree;

static void translator_mul(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto mul = std::dynamic_pointer_cast<const ov::op::v1::Multiply>(node);
    if (!mul) return;
    auto& input0 = mul->input_value(0);
    auto& input1 = mul->input_value(1);
    auto input0_name = getMLIRName(input0);
    auto input1_name = getMLIRName(input1);
    ss << "    %" << getMLIRName(node);
    if(input0.get_partial_shape().size() > 0 && input1.get_partial_shape().size() == 0) {
        ss << " = \"torch.aten.mul.Scalar\" (%" << input0_name << ", %" << input1_name << ") : ("
           << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
           << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">)";
    } else if(input0.get_partial_shape().size() == 0 && input1.get_partial_shape().size() > 0) {
        ss << " = \"torch.aten.mul.Scalar\" (%" << input1_name << ", %" << input0_name << ") : ("
           << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
           << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">)";
    } else if(input0.get_partial_shape().size() > 0 && input1.get_partial_shape().size() > 0) {
        ss << " = \"torch.aten.mul.Tensor\" (%" << input0_name << ", %" << input1_name << ") : ("
           << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
           << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">)";
    } else {
        OPENVINO_THROW("Unsupported combination");
    }
    genOutputTypes(ss, node);
    ss << "\n";
}

EMIT_REG("Multiply", translator_mul);
