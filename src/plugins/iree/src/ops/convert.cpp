#include "plugin/iree/common.hpp"
#include "openvino/op/convert.hpp"

using namespace ov::iree;

static void translator_convert(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto convert = std::dynamic_pointer_cast<const ov::op::v0::Convert>(node);
    if (!convert) return;

    auto node_input = node->input(0);
    ov::element::Type output_type = convert->get_destination_type();

    ss << "    %" << getMLIRName(node)
    << " = \"torch.aten._to_copy\" (%" << getMLIRName(node_input.get_source_output()) << ", %dtype_" << ov_to_mlir_type(output_type) << ", %none, %none, %none, %false, %none)"
    << " : (!torch.vtensor<" << node_input.get_partial_shape() << "," << ov_to_mlir_type(node_input.get_source_output().get_element_type()) << ">, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none)"
    << " -> !torch.vtensor<" << node_input.get_partial_shape() << "," << ov_to_mlir_type(output_type) << ">";

    ss << "\n";
}

EMIT_REG("Convert", translator_convert);
