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
    } else {
        output_type = shape_of_v3->get_output_type();
    }

    if(output_type == ov::element::i32) {
        ss << "    %" << getMLIRName(node)
        << " = \"torch.aten._shape_as_tensor\" ";

        genInputNames(ss, node);
        ss << " : ";
        genInputTypes(ss, node);
        genOutputTypes(ss, node);

        ss << "\n";
    } else if(output_type == ov::element::i64) {
        ss << "    %" << getMLIRName(node)
        << "_si32 = \"torch.aten._shape_as_tensor\" ";

        genInputNames(ss, node);
        ss << " : ";
        genInputTypes(ss, node);
        ss << " -> !torch.vtensor<"
           << node->get_output_partial_shape(0) << ",si32>";
        ss << "\n";

        ss << "    %" << getMLIRName(node)
        << " = \"torch.aten._to_copy\" (%" << getMLIRName(node) << "_si32, %dtype_" << ov_to_mlir_type(output_type) << ", %none, %none, %none, %false, %none)"
        << " : (!torch.vtensor<" << node->get_output_partial_shape(0) << ",si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none)"
        << " -> !torch.vtensor<" << node->get_output_partial_shape(0) << "," << ov_to_mlir_type(output_type) << ">";

        ss << "\n";
    } else {
        OPENVINO_THROW("Unsupported output_type");
    }
}

EMIT_REG("ShapeOf", translator_shape_of);
