#include "infer_request.hpp"

namespace ov {
namespace mlir {

InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& model)
    : ov::IInferRequest(model) {}

void InferRequest::infer() {}

} // namespace mlir    
} // namespace ov
