#include "infer_request.hpp"

namespace ov {
namespace mlir {

SyncInferRequest::SyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model)
    : m_compiled_model{compiled_model} {
    if (!m_compiled_model) {
        OPENVINO_THROW("Bad arguments");
    }
}

void SyncInferRequest::infer() {
    OPENVINO_THROW("Not implemented");
}

void SyncInferRequest::set_tensor(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const std::string& name) const {
    OPENVINO_THROW("Not implemented");
}

void SyncInferRequest::set_tensors(const std::string& name, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OPENVINO_THROW("Not implemented");
}

std::vector<ov::SoPtr<ov::ITensor>> SyncInferRequest::get_tensors(const std::string& name) const {
    OPENVINO_THROW("Not implemented");
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_input_tensor(size_t idx) const {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_output_tensor(size_t idx) const {
    OPENVINO_THROW("Not implemented");
}

void SyncInferRequest::set_input_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_THROW("Not implemented");
}

void SyncInferRequest::set_output_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) {
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

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model,
                                     const std::shared_ptr<SyncInferRequest>& sync_request)
    : m_compiled_model{compiled_model},
      m_sync_request{sync_request} {
    if (!m_compiled_model || !m_sync_request) {
        OPENVINO_THROW("Bad arguments");
    }
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

void AsyncInferRequest::set_tensor(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) {
    // Forward to sync request (typical design), but since this is a stub:
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> AsyncInferRequest::get_tensor(const std::string& name) const {
    OPENVINO_THROW("Not implemented");
}

void AsyncInferRequest::set_tensors(const std::string& name, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OPENVINO_THROW("Not implemented");
}

std::vector<ov::SoPtr<ov::ITensor>> AsyncInferRequest::get_tensors(const std::string& name) const {
    OPENVINO_THROW("Not implemented");
}

void AsyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> AsyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> AsyncInferRequest::get_input_tensor(size_t idx) const {
    OPENVINO_THROW("Not implemented");
}

ov::SoPtr<ov::ITensor> AsyncInferRequest::get_output_tensor(size_t idx) const {
    OPENVINO_THROW("Not implemented");
}

void AsyncInferRequest::set_input_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_THROW("Not implemented");
}

void AsyncInferRequest::set_output_tensor(size_t idx, const ov::SoPtr<ov::ITensor>& tensor) {
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
