#include "emit_reg.hpp"
#include "openvino/op/result.hpp"

using namespace ov::mlir;

static void translator_result(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto r = std::dynamic_pointer_cast<const ov::op::v0::Result>(node);
    if (!r) return;
    auto src = r->input_value(0).get_node()->get_friendly_name();
    ss << "  return %" << src << " : " << "!torch.vtensor<" << r->get_input_partial_shape(0)
           << "," << r->get_input_element_type(0).get_type_name() << ">" << "\n";
}

EMIT_REG("Result", translator_result);
