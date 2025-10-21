#pragma once
#include "openvino/runtime/iplugin.hpp"
#include <ostream>
#include <memory>

namespace ov {
namespace iree {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin() override = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(
        const std::shared_ptr<const ov::Model>& model,
        const ov::AnyMap& config) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    // Remote context compile (not implemented for this plugin)
    std::shared_ptr<ov::ICompiledModel> compile_model(
        const std::shared_ptr<const ov::Model>& model,
        const ov::AnyMap& properties,
        const ov::SoPtr<ov::IRemoteContext>& context) const override;

    // Context creation APIs (not implemented for this plugin)
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    // Import model APIs (not implemented for this plugin)
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    // Query model: by default report nothing supported
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
};

} // namespace iree
} // namespace ov
