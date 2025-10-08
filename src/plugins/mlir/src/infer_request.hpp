#pragma once
#include "openvino/runtime/iinfer_request.hpp"

namespace ov {
namespace mlir {

class InferRequest : public ov::IInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::ICompiledModel>& model);
    void infer() override;
};

} // namespace mlir   
} // namespace ov
