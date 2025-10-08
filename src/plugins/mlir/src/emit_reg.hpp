#pragma once
#include <string>
#include <map>
#include <sstream>
#include <memory>
#include <openvino/op/op.hpp>

namespace ov {
namespace mlir {

using translator_func = void(*)(const std::shared_ptr<const ov::Node>& node, std::stringstream& ss);

void register_translator(const std::string& op_type, translator_func func);
std::map<std::string, translator_func>& get_translators();

#define EMIT_REG(OP_TYPE, FUNC)                                   \
    static struct EmitReg_##FUNC {                                 \
        EmitReg_##FUNC() { register_translator(OP_TYPE, FUNC); }         \
    } _emit_reg_instance_##FUNC;

}
}
