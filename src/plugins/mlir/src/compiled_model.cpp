#include "compiled_model.hpp"
#include "infer_request.hpp"

namespace ov {
namespace mlir {

CompiledModel::CompiledModel(
    const std::shared_ptr<const ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::ICompiledModel(model, plugin) {}

std::shared_ptr<ov::IInferRequest> CompiledModel::create_infer_request() const {
    return std::make_shared<InferRequest>(shared_from_this());
}

void CompiledModel::export_model(std::ostream& stream) const {}

} // namespace my_plugin
} // namespace ov
