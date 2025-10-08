#include "emit_reg.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov::mlir;

static void translator_mul(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto mul = std::dynamic_pointer_cast<const ov::op::v1::Multiply>(node);
    if (!mul) return;
    auto lhs = mul->input_value(0).get_node()->get_friendly_name();
    auto rhs = mul->input_value(1).get_node()->get_friendly_name();
    ss << "  %" << mul->get_friendly_name() << " = torch.aten.mul %"
       << lhs << ", %" << rhs << "\n";
}

EMIT_REG("Multiply", translator_mul);
