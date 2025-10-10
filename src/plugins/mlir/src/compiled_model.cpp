#include "plugin/mlir/compiled_model.hpp"
#include "plugin/mlir/infer_request.hpp"
#include "plugin/mlir/common.hpp"

namespace ov {
namespace mlir {

static std::mutex translators_mutex;
static std::map<std::string, std::string> ov_to_aten = {
    {"Abs", "torch.aten.abs"},
    {"Add", "torch.aten.add"},
    {"Multiply", "torch.aten.mul"},
    {"Subtract", "torch.aten.sub"},
    {"Divide", "torch.aten.div"},
    {"Power", "torch.aten.pow"},
    {"Minimum", "torch.aten.min"},
    {"Maximum", "torch.aten.max"},
    {"Equal", "torch.aten.eq"},
    {"NotEqual", "torch.aten.ne"},
    {"Greater", "torch.aten.gt"},
    {"GreaterEqual", "torch.aten.ge"},
    {"Less", "torch.aten.lt"},
    {"LessEqual", "torch.aten.le"},
    {"LogicalAnd", "torch.aten.logical_and"},
    {"LogicalOr", "torch.aten.logical_or"},
    {"LogicalXor", "torch.aten.logical_xor"},
    {"LogicalNot", "torch.aten.logical_not"},
    {"Negative", "torch.aten.neg"},
    {"Sign", "torch.aten.sign"},
    {"Floor", "torch.aten.floor"},
    {"Ceiling", "torch.aten.ceil"},
    {"Round", "torch.aten.round"},
    {"Trunc", "torch.aten.trunc"},
    {"Clamp", "torch.aten.clamp"},
    {"Sqrt", "torch.aten.sqrt"},
    {"Exp", "torch.aten.exp"},
    {"Log", "torch.aten.log"},
    {"Sin", "torch.aten.sin"},
    {"Cos", "torch.aten.cos"},
    {"Tan", "torch.aten.tan"},
    {"Asin", "torch.aten.asin"},
    {"Acos", "torch.aten.acos"},
    {"Atan", "torch.aten.atan"},
    {"Sinh", "torch.aten.sinh"},
    {"Cosh", "torch.aten.cosh"},
    {"Tanh", "torch.aten.tanh"},
    {"Asinh", "torch.aten.asinh"},
    {"Acosh", "torch.aten.acosh"},
    {"Atanh", "torch.aten.atanh"},
    {"Erf", "torch.aten.erf"},
    {"Relu", "torch.aten.relu"},
};

std::map<std::string, translator_func>& get_translators() {
    static std::map<std::string, translator_func> translators_map;
    return translators_map;
}

bool register_translator(const std::string& op_type, translator_func func) {
    std::lock_guard<std::mutex> lock(translators_mutex);
    get_translators()[op_type] = func;
    return true;
}

void translator_ov_to_aten(const std::shared_ptr<const ov::Node>& node, std::ostream& ss, const std::string& torch_name) {
    ss << "    %" << node->get_friendly_name()
       << " = \"" << torch_name << "\" ";

    genInputNames(ss, node);
    ss << " : ";
    genInputTypes(ss, node);
    genOutputTypes(ss, node);

    ss << "\n";
}

CompiledModel::CompiledModel(
    const std::shared_ptr<const ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::ICompiledModel(model, plugin),
    m_model(model) {} // @todo need to copy model

CompiledModel::~CompiledModel() {}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    return m_model;
}

void CompiledModel::export_model(std::ostream& stream) const {
    auto& ss = stream;
    ss << "module @" << m_model->get_friendly_name() << " {\n";

    ss << "  func.func @" << m_model->get_friendly_name() << "(";

    std::string delimeter = "";
    for(auto& arg: inputs()) {
        ss << delimeter << "%" << arg.get_any_name() << ": !torch.vtensor<" << arg.get_partial_shape()
           << "," << arg.get_element_type().get_type_name() << ">";
        delimeter = ", ";
    }

    ss << ") -> ";

    delimeter = "";
    for(auto& res: outputs()) {
        ss << delimeter << "!torch.vtensor<" << res.get_partial_shape()
           << "," << res.get_element_type().get_type_name() << ">";
        delimeter = ", ";
    }

    ss << " {\n";

    auto& translators = get_translators();

    for (const auto& node : m_model->get_ordered_ops()) {
        const auto& type_info = node->get_type_info();
        // Direct translators from OV to ATen
        {
            // Find version-specific translator "Abs::opset1", and if it isn't found - common "Abs"
            auto it = ov_to_aten.find(std::string(type_info.name) + "::" + type_info.version_id);
            if(it == ov_to_aten.end()) {
                it = ov_to_aten.find(type_info.name);
            }
            it = ov_to_aten.find(type_info.name);
            if (it != ov_to_aten.end()) {
                std::cout << "Found simple translator for " << type_info.name << " to " << it->second << std::endl;
                translator_ov_to_aten(node, ss, it->second);
                continue;
            }
        }
        // Custom translators from OV to ATen
        {
            // Find version-specific translator "Abs::opset1", and if it isn't found - common "Abs"
            auto it = translators.find(std::string(type_info.name) + "::" + type_info.version_id);
            if(it == translators.end()) {
                it = translators.find(type_info.name);
            }
            if (it != translators.end()) {
                auto fn = it->second;
                fn(node, ss);
                continue;
            }
        }
        // Failed to translate operation
        {
            ss << "  // Unsupported node: " << node->get_friendly_name()
               << " (" << type_info.name << ")\n";
        }
    }
    ss << "  }\n"; // func.func
    ss << "}\n"; // module
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    (void)name;
    OPENVINO_THROW("Not implemented");
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    (void)properties;
    OPENVINO_THROW("Not implemented");
}

} // namespace mlir
} // namespace ov
