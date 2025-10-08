#pragma once
#include <string>
#include <map>
#include <ostream>
#include <memory>
#include <openvino/op/op.hpp>

namespace ov {
namespace mlir {

using translator_func = void(*)(const std::shared_ptr<const ov::Node>& node, std::ostream& ss);

bool register_translator(const std::string& op_type, translator_func func);
std::map<std::string, translator_func>& get_translators();

#define EMIT_REG(OP_TYPE, FUNC) static bool translator_reg_##FUNC = register_translator(OP_TYPE, FUNC);

}
}
