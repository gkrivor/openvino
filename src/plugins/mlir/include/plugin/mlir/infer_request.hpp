#pragma once
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "openvino/core/except.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

// Forward declaration
struct iree_hal_device_t;
struct iree_runtime_session_t;

namespace ov {
namespace mlir {

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    // Construct with the owning compiled model and its input/output ports if needed.
    explicit SyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model);

    // --- ISyncInferRequest interface ---
    void infer() override;

    // By port (for newer APIs using Output<Node>)
    // void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    // ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    // State & profiling (commonly part of the interface)
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

protected:
    // Helpers often used by derived classes (no-op stubs here)
    void check_tensors() const;

private:
    std::shared_ptr<const ov::ICompiledModel> m_compiled_model;

    // Basic containers for inputs/outputs; real plugin would keep backend-specific handles
    std::map<std::string, ov::SoPtr<ov::ITensor>> m_tensors_by_name;
    std::vector<ov::SoPtr<ov::ITensor>> m_input_tensors;
    std::vector<ov::SoPtr<ov::ITensor>> m_output_tensors;
};

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);
    ~AsyncInferRequest() override;

    // --- IAsyncInferRequest interface ---
    void infer() override;
    void start_async() override;
    void wait() override;                 // wait indefinitely
    bool wait_for(const std::chrono::milliseconds& timeout) override;
    void cancel() override;

    // void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    // ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    std::shared_ptr<const ov::ICompiledModel> m_compiled_model;
    std::shared_ptr<SyncInferRequest> m_sync_request;
    iree_hal_device_t* m_iree_device;
    iree_runtime_session_t *m_iree_session;
};

} // namespace mlir   
} // namespace ov
