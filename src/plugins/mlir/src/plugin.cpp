#include "plugin.hpp"
#include "compiled_model.hpp"
#include "emit_reg.hpp"

#include <iostream>
#include <mutex>

namespace ov {
namespace mlir {

static std::map<std::string, translator_func> translators_map;
static std::mutex translators_mutex;

std::map<std::string, emit_func>& get_translators() {
    return translators_map;
}

void register_emit(const std::string& op_type, translator_func func) {
    std::lock_guard<std::mutex> lock(translators_mutex);
    translators_map[op_type] = func;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& config) const {
    return std::make_shared<CompiledModel>(model, shared_from_this());
}

void Plugin::model_to_mlir(const std::shared_ptr<const ov::Model>& model,
                           const std::shared_ptr<std::stringstream>& out) const {
    auto& ss = *out;
    ss << "module @" << model->get_friendly_name() << " {\n";

    auto& translators = get_translators();

    for (const auto& node : model->get_ordered_ops()) {
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

} // namespace mlir
} // namesppace ov

extern "C" OPENVINO_PLUGIN_API ov::IPlugin* CreatePluginEngine() {
    return new ov::mlir::Plugin();
}
