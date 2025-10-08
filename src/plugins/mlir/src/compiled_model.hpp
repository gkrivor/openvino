#pragma once
#include "openvino/runtime/icompiled_model.hpp"

namespace ov {
namespace mlir {

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    void export_model(std::ostream& stream) const override;
};

} // namespace my_plugin
} // namespace ov
