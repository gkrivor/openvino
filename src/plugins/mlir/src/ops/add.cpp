#include "emit_reg.hpp"
#include "openvino/op/add.hpp"

static void emit_add(const std::shared_ptr<const ov::Node>& node, std::stringstream& ss) {
    auto add = std::dynamic_pointer_cast<const ov::op::v1::Add>(node);
    if (!add) return;
    auto lhs = add->input_value(0).get_node()->get_friendly_name();
    auto rhs = add->input_value(1).get_node()->get_friendly_name();
    ss << "  %" << add->get_friendly_name() << " = torch.aten.add %"
       << lhs << ", %" << rhs << "\n";
}

EMIT_REG("Add", emit_add);
