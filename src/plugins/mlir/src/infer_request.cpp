#include "infer_request.hpp"

namespace ov {
namespace mlir {

SyncInferRequest::SyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
    m_compiled_model{compiled_model} {
    if (!m_compiled_model) {
        OPENVINO_THROW("Bad arguments");
    }
}

void SyncInferRequest::infer() {
    OPENVINO_THROW("Not implemented");
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OPENVINO_THROW("Not implemented");
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    OPENVINO_THROW("Not implemented");
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    OPENVINO_THROW("Not implemented");
}

void SyncInferRequest::check_tensors() const {
    OPENVINO_THROW("Not implemented");
}

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(infer_request, task_executor, callback_executor)
    , m_sync_request(infer_request) {
}

void AsyncInferRequest::start_async() {
    OPENVINO_THROW("Not implemented");
}

void AsyncInferRequest::wait() {
    OPENVINO_THROW("Not implemented");
}

bool AsyncInferRequest::wait_for(const std::chrono::milliseconds& /*timeout*/) {
    OPENVINO_THROW("Not implemented");
}

void AsyncInferRequest::cancel() {
    OPENVINO_THROW("Not implemented");
}

void AsyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> AsyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OPENVINO_THROW("Not implemented");
}

std::vector<ov::SoPtr<ov::IVariableState>> AsyncInferRequest::query_state() const {
    OPENVINO_THROW("Not implemented");
}

std::vector<ov::ProfilingInfo> AsyncInferRequest::get_profiling_info() const {
    OPENVINO_THROW("Not implemented");
}

} // namespace mlir    
} // namespace ov
