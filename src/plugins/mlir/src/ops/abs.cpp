#include "common.hpp"
#include "openvino/op/abs.hpp"

using namespace ov::mlir;

static void translator_abs(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto abs = std::dynamic_pointer_cast<const ov::op::v0::Abs>(node);
    if (!abs) return;
    auto lhs = abs->input_value(0).get_node()->get_friendly_name();
    ss << "  %" << abs->get_friendly_name() << " = \"torch.aten.abs\" (%"
       << lhs << ") : (";

    std::string delimeter = "";
    for(auto& arg : abs->inputs()) {
        ss << delimeter << "!torch.vtensor<" << arg.get_partial_shape()
           << "," << arg.get_element_type().get_type_name() << ">";
        delimeter = ", ";
    }

    ss << ") -> ";

    delimeter = "";
    for(auto& res : abs->outputs()) {
        ss << delimeter << "!torch.vtensor<" << res.get_partial_shape()
           << "," << res.get_element_type().get_type_name() << ">";
        delimeter = ", ";
    }
    ss << "\n";
}

EMIT_REG("Abs", translator_abs);
