#pragma once
#include "openvino/runtime/icompiled_model.hpp"

namespace ov {

// Forward declarations
class Model;
class Node;
class Output;

namespace mlir {

class CompiledModel final : public ov::ICompiledModel {
public:
    CompiledModel();
    ~CompiledModel() override;

    // Create synchronous or async infer request (depending on your dev API)
    // If your ICompiledModel has only one of these, remove the other.
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<ov::ISyncInferRequest>  create_sync_infer_request() const override;

    // Model shape/graph reflection
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    // Input/Output ports (const versions in dev API)
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    // Named single-port accessors
    ov::Output<const ov::Node> input(const std::string& tensor_name) const override;
    ov::Output<const ov::Node> output(const std::string& tensor_name) const override;

    // Binary export
    void export_model(std::ostream& stream) const override;

    // Properties (RW)
    ov::Any get_property(const std::string& name) const override;
    void set_property(const ov::AnyMap& properties) override;

};


} // namespace mlir
} // namespace ov
