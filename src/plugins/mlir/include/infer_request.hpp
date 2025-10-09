#pragma once
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "openvino/core/except.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/state.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {
namespace mlir {

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    // Construct with the owning compiled model and its input/output ports if needed.
    explicit SyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model);

    // --- ISyncInferRequest interface ---
    void infer() override;

    // Tensors by port/index/name – depending on what your plugin exposes.
    // Keep all common overloads implemented to satisfy pure virtuals across OV versions.

    // By tensor name
    void set_tensor(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) override;
    ov::SoPtr<ov::ITensor> get_tensor(const std::string& name) const override;

    // Batch set/get by name (optional in some versions; keep for completeness)
    void set_tensors(const std::string& name, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const std::string& name) const override;

    // By port (for newer APIs using Output<Node>)
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    // Convenience input/output accessors (commonly used)
    ov::SoPtr<ov::ITensor> get_input_tensor(size_t idx) const override;
    ov::SoPtr<ov::ITensor> get_output_tensor(size_t idx) const override;
    void set_input_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) override;
    void set_output_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) override;

    // State & profiling (commonly part of the interface)
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

protected:
    // Helpers often used by derived classes (no-op stubs here)
    void check_tensors() const;

private:
    std::shared_ptr<ov::ICompiledModel> m_compiled_model;

    // Basic containers for inputs/outputs; real plugin would keep backend-specific handles
    std::map<std::string, ov::SoPtr<ov::ITensor>> m_tensors_by_name;
    std::vector<ov::SoPtr<ov::ITensor>> m_input_tensors;
    std::vector<ov::SoPtr<ov::ITensor>> m_output_tensors;
};

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    // Async IR typically wraps a sync IR and an executor(s)
    AsyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model,
                      const std::shared_ptr<SyncInferRequest>& sync_request);

    // --- IAsyncInferRequest interface ---
    void start_async() override;
    void wait() override;                 // wait indefinitely
    bool wait_for(const std::chrono::milliseconds& timeout) override;
    void cancel() override;

    // Forward tensor/state/profiling access to the underlying sync IR:
    void set_tensor(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) override;
    ov::SoPtr<ov::ITensor> get_tensor(const std::string& name) const override;

    void set_tensors(const std::string& name, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const std::string& name) const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    ov::SoPtr<ov::ITensor> get_input_tensor(size_t idx) const override;
    ov::SoPtr<ov::ITensor> get_output_tensor(size_t idx) const override;
    void set_input_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) override;
    void set_output_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    std::shared_ptr<ov::ICompiledModel> m_compiled_model;
    std::shared_ptr<SyncInferRequest> m_sync_request;
};

} // namespace mlir   
} // namespace ov
