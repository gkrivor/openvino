#include "plugin/iree/common.hpp"
#include "openvino/op/matmul.hpp"

using namespace ov::iree;

static void translator_matmul(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto matmul = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node);
    if (!matmul) return;
    auto& input0 = matmul->input_value(0);
    auto& input1 = matmul->input_value(1);
    auto input0_name = getMLIRName(input0);
    auto input1_name = getMLIRName(input1);
    if(matmul->get_transpose_a()) {
        // @todo: Handle transpose first input
        OPENVINO_THROW("Transposed A input isn't supported");
    }
    if(matmul->get_transpose_b()) {
        // @todo: Handle transpose second input
        OPENVINO_THROW("Transposed B input isn't supported");
    }
    ss << "    %" << getMLIRName(node);
    ss << " = \"torch.aten.matmul\" (%" << input0_name << ", %" << input1_name << ") : ("
        << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
        << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">)";
    genOutputTypes(ss, node);
    ss << "\n";
}

EMIT_REG("MatMul", translator_matmul);
