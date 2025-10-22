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
namespace iree {

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    // Construct with the owning compiled model and its input/output ports if needed.
    explicit SyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model);

    // --- ISyncInferRequest interface ---
    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

protected:
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

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    std::shared_ptr<const ov::ICompiledModel> m_compiled_model;
    std::shared_ptr<SyncInferRequest> m_sync_request;
    iree_hal_device_t* m_iree_device;
    iree_runtime_session_t *m_iree_session;
};

} // namespace iree   
} // namespace ov
