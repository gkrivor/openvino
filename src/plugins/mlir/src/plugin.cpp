#include "plugin.hpp"
#include "compiled_model.hpp"
#include "emit_reg.hpp"

#include <fstream>
#include <iostream>
#include <mutex>

namespace ov {
namespace mlir {

static std::map<std::string, translator_func> translators_map;
static std::mutex translators_mutex;

std::map<std::string, translator_func>& get_translators() {
    return translators_map;
}

bool register_translator(const std::string& op_type, translator_func func) {
    std::lock_guard<std::mutex> lock(translators_mutex);
    translators_map[op_type] = func;
    return true;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& config) const {
    const auto output_file = std::make_shared<std::ofstream>("debug.mlir", std::ios::binary);
    model_to_mlir(model, output_file);
    return std::make_shared<CompiledModel>(model, shared_from_this());
}

void Plugin::set_property(const ov::AnyMap& properties) {
    (void)properties;
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    (void)name;
    (void)arguments;
    return {};
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    (void)model;
    (void)properties;
    (void)context;
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    (void)remote_properties;
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    (void)remote_properties;
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model, const ov::AnyMap& properties) const {
    (void)model;
    (void)properties;
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    (void)model;
    (void)context;
    (void)properties;
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model, const ov::AnyMap& properties) const {
    (void)model;
    (void)properties;
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    (void)model;
    (void)context;
    (void)properties;
    OPENVINO_THROW("Not implemented");
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    (void)model;
    (void)properties;
    return {};
}

void Plugin::model_to_mlir(const std::shared_ptr<const ov::Model>& model,
                           const std::shared_ptr<std::ostream>& out) const {
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

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_template_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::mlir::Plugin, version)
// ! [plugin:create_plugin_engine]
