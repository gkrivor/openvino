#include "plugin/mlir/plugin.hpp"
#include "plugin/mlir/compiled_model.hpp"
#include "openvino/runtime/internal_properties.hpp"

#include <fstream>
#include <iostream>
#include <mutex>

namespace ov {
namespace mlir {

static std::vector<ov::PropertyName> supported_configKeys = {};

Plugin::Plugin() {
    set_device_name("MLIR");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& config) const {
    return compile_model(model, config, {});
}

void Plugin::set_property(const ov::AnyMap& properties) {
    (void)properties;
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name)) {
        OPENVINO_THROW("The Value is not set for ", name);
    } else if (name == ov::supported_properties.name()) {
        std::vector<ov::PropertyName> property_name;
        property_name.push_back(ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO});
        property_name.push_back(ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO});
        for (auto& it : supported_configKeys) {
            property_name.push_back(it);
        }
        return decltype(ov::supported_properties)::value_type(std::move(property_name));
    } else if (name == ov::internal::supported_properties.name()) {
        return decltype(ov::internal::supported_properties)::value_type{};
    } else if (name == ov::device::full_name.name()) {
        return get_device_name();
    } else {
        OPENVINO_THROW("Unsupported property: ", name);
    }}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    (void)properties;
    (void)context;
    return std::make_shared<CompiledModel>(model, shared_from_this());
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

} // namespace mlir
} // namesppace ov

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_mlir_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::mlir::Plugin, version)
// ! [plugin:create_plugin_engine]
