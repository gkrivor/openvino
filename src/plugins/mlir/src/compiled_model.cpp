#include "compiled_model.hpp"
#include "infer_request.hpp"

namespace ov {
namespace mlir {

CompiledModel::CompiledModel(
    const std::shared_ptr<const ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::ICompiledModel(model, plugin) {}

CompiledModel::~CompiledModel() {}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW("Not implemented");
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    OPENVINO_THROW("Not implemented");
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::inputs() const {
    OPENVINO_THROW("Not implemented");
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::outputs() const {
    OPENVINO_THROW("Not implemented");
}

void CompiledModel::export_model(std::ostream& stream) const {
    (void)stream;
    OPENVINO_THROW("Not implemented");
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
