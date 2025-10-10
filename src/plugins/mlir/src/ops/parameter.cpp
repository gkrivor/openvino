#include "plugin/mlir/common.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov::mlir;

static void translator_parameter(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto p = std::dynamic_pointer_cast<const ov::op::v0::Parameter>(node);
    if (!p) return;
/*
    ss << "  %" << p->get_friendly_name()
       << " = torch.parameter : tensor<" << p->get_output_partial_shape(0)
       << "x" << p->get_output_element_type(0).get_type_name() << ">\n";
*/
}

EMIT_REG("Parameter", translator_parameter);
