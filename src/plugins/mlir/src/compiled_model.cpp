#include "compiled_model.hpp"
#include "infer_request.hpp"
#include "common.hpp"

namespace ov {
namespace mlir {

static std::mutex translators_mutex;
static std::map<std::string, std::string> ov_to_aten = {
    {"Abs", "torch.aten.abs"},
    {"Add", "torch.aten.add"},
    {"Mul", "torch.aten.mul"},
    {"Sub", "torch.aten.sub"}
}

std::map<std::string, translator_func>& get_translators() {
    static std::map<std::string, translator_func> translators_map;
    return translators_map;
}

bool register_translator(const std::string& op_type, translator_func func) {
    std::lock_guard<std::mutex> lock(translators_mutex);
    get_translators()[op_type] = func;
    return true;
}

void direct_translator(const std::shared_ptr<const ov::Node>& node, std::ostream& ss, const std::string& torch_name) {
    ss << " %" << node->get_friendly_name()
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
        auto type_name = node->get_type_name();
        // Direct translators from OV to ATen
        {
            auto& it = ov_to_aten.find(type_name);
            if (it != ov_to_aten.end()) {
                direct_translator(node, ss, it.second);
                continue;
            }
        }
        // Custom translators from OV to ATen
        {
            auto& it = translators.find(type_name);
            if (it != translators.end()) {
                auto fn = it->second;
                fn(node, ss);
                continue;
            }
        }
        // Failed to translate operation
        {
            ss << "  // Unsupported node: " << node->get_friendly_name()
               << " (" << type_name << ")\n";
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
