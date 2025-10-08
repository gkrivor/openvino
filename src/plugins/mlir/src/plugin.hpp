#pragma once
#include "openvino/runtime/iplugin.hpp"
#include <sstream>
#include <memory>

namespace ov {
namespace mlir {

class Plugin : public ov::IPlugin {
public:
    Plugin() = default;
    ~Plugin() override = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(
        const std::shared_ptr<const ov::Model>& model,
        const ov::AnyMap& config) const override;

    void set_property(const ov::AnyMap& properties) override {}
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override { return {}; }

    void model_to_mlir(const std::shared_ptr<const ov::Model>& model,
                       const std::shared_ptr<std::stringstream>& out) const;
};

} // namespace mlir
} // namespace ov
