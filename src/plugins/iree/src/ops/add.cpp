#include "plugin/iree/common.hpp"
#include "openvino/op/add.hpp"

using namespace ov::iree;

static void translator_add(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto add = std::dynamic_pointer_cast<const ov::op::v1::Add>(node);
    if (!add) return;
    auto& input0 = add->input_value(0);
    auto& input1 = add->input_value(1);
    auto input0_name = getMLIRName(input0);
    auto input1_name = getMLIRName(input1);
    ss << "    %" << getMLIRName(node);
    if(input0.get_partial_shape().size() > 0 && input1.get_partial_shape().size() == 0) {
        ss << " = \"torch.aten.add.Scalar\" (%" << input0_name << ", %" << input1_name << ", %const_one) : ("
           << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
           << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
           << "!torch.int)";
    } else if(input0.get_partial_shape().size() == 0 && input1.get_partial_shape().size() > 0) {
        ss << " = \"torch.aten.add.Scalar\" (%" << input1_name << ", %" << input0_name << ", %const_one) : ("
           << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
           << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
           << "!torch.int)";
    } else if(input0.get_partial_shape().size() > 0 && input1.get_partial_shape().size() > 0) {
        ss << " = \"torch.aten.add.Tensor\" (%" << input0_name << ", %" << input1_name << ", %const_one) : ("
           << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
           << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
           << "!torch.int)";
    } else {
        OPENVINO_THROW("Unsupported combination");
    }
    genOutputTypes(ss, node);
    ss << "\n";
}

EMIT_REG("Add", translator_add);
