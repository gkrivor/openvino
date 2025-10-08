#include "emit_reg.hpp"
#include "openvino/op/abs.hpp"

using namespace ov::mlir;

static void translator_abs(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto abs = std::dynamic_pointer_cast<const ov::op::v0::Abs>(node);
    if (!abs) return;
    auto lhs = abs->input_value(0).get_node()->get_friendly_name();
    ss << "  %" << abs->get_friendly_name() << " = torch.aten.abs %"
       << lhs << "\n";
}

EMIT_REG("Abs", translator_abs);
