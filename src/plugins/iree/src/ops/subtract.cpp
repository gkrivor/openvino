#include "plugin/iree/common.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::iree;

static void translator_subtract(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto sub = std::dynamic_pointer_cast<const ov::op::v1::Subtract>(node);
    if (!sub) return;
    auto& input0 = sub->input_value(0);
    auto& input1 = sub->input_value(1);
    auto input0_name = getMLIRName(input0);
    auto input1_name = getMLIRName(input1);
    ss << "    %" << getMLIRName(node);
    if(input0.get_partial_shape().size() > 0 && input1.get_partial_shape().size() > 0) {
        ss << " = \"torch.aten.sub.Tensor\" (%" << input0_name << ", %" << input1_name << ", %const_one) : ("
           << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
           << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
           << "!torch.int)";
    } else {
        OPENVINO_THROW("Unsupported combination");
    }
    genOutputTypes(ss, node);
    ss << "\n";
}

EMIT_REG("Subtract", translator_subtract);
