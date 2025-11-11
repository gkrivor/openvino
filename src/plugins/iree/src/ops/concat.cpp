#include "plugin/iree/common.hpp"
#include "openvino/op/concat.hpp"

using namespace ov::iree;

static void translator_concat(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto concat = std::dynamic_pointer_cast<const ov::op::v0::Concat>(node);
    if (!concat) return;
    //%tensors0 = "torch.prim.ListConstruct" (%tensors0_v) : (!torch.vtensor<[1], f32>) -> !torch.list<vtensor<[1], f32>>
    //%dim1 = torch.constant.int 0
    ss << "    %" << getMLIRName(node) << "_tensors = \"torch.prim.ListConstruct\" (";
    auto& inputs = concat->inputs();
    std::string delimiter = "";
    for(auto& item: inputs) {
        ss << delimiter << "%" << getMLIRName(item);
        delimiter = ", ";
    }
    ss << ") : (";
    delimiter = "";
    for(auto& item: inputs) {
        ss << delimiter << "!torch.vtensor<" << item.get_partial_shape() << "," << ov_to_mlir_type(item.get_element_type()) << ">";
        delimiter = ", ";
    }
    std::string node_name = getMLIRName(node);
    ss << ") -> !torch.list<vtensor>\n";
    ss << "    %" << node_name << "_dim = torch.constant.int " << concat->get_axis() << "\n";
    ss << "    %" << node_name << " = \"torch.aten.cat\" (%" << node_name << "_tensors, %" << node_name << "_dim) : (!torch.list<vtensor>, !torch.int)";
    genOutputTypes(ss, node);
    ss << "\n";
}

EMIT_REG("Concat", translator_concat);
