#pragma once
#include "openvino/runtime/icompiled_model.hpp"

namespace ov {

// Forward declarations
class Model;
class Node;

namespace mlir {

class CompiledModel final : public ov::ICompiledModel {
private:
    std::shared_ptr<const ov::Model> m_model;
    uint8_t *m_compiled;
    size_t m_compiled_size;
    std::string m_module_name;

public:
    CompiledModel(
        const std::shared_ptr<const ov::Model>& model,
        const std::shared_ptr<const ov::IPlugin>& plugin);
    ~CompiledModel() override;

    void init();

    // Create synchronous or async infer request (depending on your dev API)
    // If your ICompiledModel has only one of these, remove the other.
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<ov::ISyncInferRequest>  create_sync_infer_request() const override;

    // Model shape/graph reflection
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    // Binary export
    void export_model(std::ostream& stream) const override;

    // Properties (RW)
    ov::Any get_property(const std::string& name) const override;
    void set_property(const ov::AnyMap& properties) override;

    const uint8_t* get_compiled_mlir() const {
        return m_compiled;
    }
    size_t get_compiled_mlir_size() const {
        return m_compiled_size;
    }
    std::string get_module_name() const {
        return m_module_name;
    }

protected:
    void generate_mlir(std::ostream& stream) const;
    void reset_compiled();
};


} // namespace mlir
} // namespace ov
