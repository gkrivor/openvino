#include "plugin/iree/common.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov::iree;

static void translator_shape_of(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    ov::element::Type output_type = ov::element::i64;
    auto shape_of_v3 = std::dynamic_pointer_cast<const ov::op::v3::ShapeOf>(node);
    if (!shape_of_v3) {
        auto shape_of_v1 = std::dynamic_pointer_cast<const ov::op::v0::ShapeOf>(node);
        if(!shape_of_v1) {
            return;
        }
    };
    auto node_input = node->input(0);

    if(output_type == ov::element::i32) {
        ss << "    %" << getMLIRName(node)
        << " = \"torch.aten._shape_as_tensor\" ";

        genInputNames(ss, node);
        ss << " : ";
        genInputTypes(ss, node);
        genOutputTypes(ss, node);

        ss << "\n";
    } else {
        ss << "    %" << getMLIRName(node)
        << "_si32 = \"torch.aten._shape_as_tensor\" ";

        genInputNames(ss, node);
        ss << " : ";
        genInputTypes(ss, node);
        genOutputTypes(ss, node);
        ss << "\n";

        ss << "    %" << getMLIRName(node)
        << " = \"torch.aten._to_copy\" (%" << getMLIRName(node) << "_si32, %dtype_" << ov_to_mlir_type(output_type) << ", %none, %none, %none, %false, %none)"
        << " : (!torch.vtensor<" << node_input.get_partial_shape() << ",si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none)"
        << " -> !torch.vtensor<" << node_input.get_partial_shape() << "," << ov_to_mlir_type(output_type) << ">";

        ss << "\n";
    }
}

EMIT_REG("ShapeOf", translator_shape_of);
