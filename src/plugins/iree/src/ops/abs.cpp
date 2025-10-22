#include "plugin/iree/common.hpp"
#include "openvino/op/abs.hpp"

using namespace ov::iree;

static void translator_abs(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto abs = std::dynamic_pointer_cast<const ov::op::v0::Abs>(node);
    if (!abs) return;

    ss << " %" << abs->get_friendly_name()
       << " = \"torch.aten.abs\" ";

    genInputNames(ss, node);
    ss << " : ";
    genInputTypes(ss, node);
    genOutputTypes(ss, node);

    ss << "\n";
}

EMIT_REG("Abs", translator_abs);
