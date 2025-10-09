#include "compiled_model.hpp"
#include "infer_request.hpp"
#include "emit_reg.hpp"

namespace ov {
namespace mlir {

static std::mutex translators_mutex;

std::map<std::string, translator_func>& get_translators() {
    static std::map<std::string, translator_func> translators_map;
    return translators_map;
}

bool register_translator(const std::string& op_type, translator_func func) {
    std::lock_guard<std::mutex> lock(translators_mutex);
    get_translators()[op_type] = func;
    return true;
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

    auto& translators = get_translators();

    for (const auto& node : m_model->get_ordered_ops()) {
        auto type_name = node->get_type_name();
        auto it = translators.find(type_name);
        if (it != translators.end()) {
            auto fn = it->second;
            fn(node, ss);
        } else {
            ss << "  // Unsupported node: " << node->get_friendly_name()
               << " (" << type_name << ")\n";
        }
    }
    ss << "}\n";
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
