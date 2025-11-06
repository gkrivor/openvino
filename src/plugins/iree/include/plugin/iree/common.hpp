#pragma once
#include <string>
#include <map>
#include <ostream>
#include <memory>
#include <openvino/op/op.hpp>

namespace ov {
namespace iree {

/* ================ Translators Registration Processing ================ */
using translator_func = void(*)(const std::shared_ptr<const ov::Node>& node, std::ostream& ss);

bool register_translator(const std::string& op_type, translator_func func, bool is_post_processor = false);
std::map<std::string, translator_func>& get_translators();

#define EMIT_REG(OP_TYPE, FUNC) static bool translator_reg_##FUNC = register_translator(OP_TYPE, FUNC);
#define EMIT_POST(OP_TYPE, FUNC) static bool post_processor_reg_##FUNC = register_translator(OP_TYPE, FUNC, true);

/* ================ MLIR generators ================ */
void genInputNames(std::ostream& ss, const std::shared_ptr<const ov::Node>& node);
void genInputTypes(std::ostream& ss, const std::shared_ptr<const ov::Node>& node);
void genOutputTypes(std::ostream& ss, const std::shared_ptr<const ov::Node>& node);

/* ================ MLIR Naming Conversion ================ */
std::string getMLIRName(const ov::Node* node);
std::string getMLIRName(const std::shared_ptr<const ov::Node>& node);
std::string getMLIRName(const ov::Output<const ov::Node>& node);

/* ================ OpenVINO Type Converters ================ */
std::string ov_to_mlir_type(ov::element::Type t);
uint8_t ov_to_torch_dtype(ov::element::Type t);

} // namespace mlir
} // namespace ov
