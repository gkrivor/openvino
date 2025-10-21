#include "plugin/mlir/infer_request.hpp"
#pragma warning(push)
#pragma warning(disable : 4146)
#pragma warning(disable : 4200)
#include "iree/runtime/api.h"
#pragma warning(pop)
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace mlir {

namespace {

void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor, const ov::element::Type& element_type, const ov::Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::make_tensor(element_type, shape);
    } else {
        tensor->set_shape(shape);
    }
}

}  // namespace

namespace iree {
static iree_runtime_instance_t* instance = NULL;

static void init() {
    static bool is_initialized = false;
    if(is_initialized) {
        return;
    }
    static iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);
    iree_status_t status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &instance);
    // @todo: think how to call a GlobalShutdown...
    is_initialized = true;
}
}

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
    m_compiled_model{compiled_model} {
    if (!m_compiled_model) {
        OPENVINO_THROW("Bad arguments");
    }
    iree::init();
    // Allocate input/output tensors
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                input.get_element_type(),
                                input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                output.get_element_type(),
                                output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
}

void SyncInferRequest::infer() {
    OPENVINO_NOT_IMPLEMENTED;
}
/*
void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OPENVINO_NOT_IMPLEMENTED;
}
*/
std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void SyncInferRequest::check_tensors() const {
    // 1st call on infer()
    // Let's imagine all tensors are fine
    // OPENVINO_NOT_IMPLEMENTED;
}

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(infer_request, task_executor, callback_executor)
    , m_sync_request(infer_request)
    , m_iree_device(nullptr)
    , m_iree_session(nullptr) {
}

AsyncInferRequest::~AsyncInferRequest() {
    if(m_iree_device) {
        iree_hal_device_release(m_iree_device);
    }
    m_iree_device = nullptr;
}

void AsyncInferRequest::infer() {
    iree_status_t status = nullptr;
    status = iree_runtime_instance_try_create_default_device(iree::instance, iree_make_cstring_view("local-task"), &m_iree_device);
    if(status) {
        OPENVINO_THROW("Error creating iree device");
    }
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    iree_runtime_session_t* session = NULL;
    status = iree_runtime_session_create_with_device(
      iree::instance, &session_options, m_iree_device,
      iree_runtime_instance_host_allocator(iree::instance), &session);
    if(status) {
        OPENVINO_THROW("Error creating iree session");
    }
}

void AsyncInferRequest::start_async() {
    OPENVINO_NOT_IMPLEMENTED;
}

void AsyncInferRequest::wait() {
    // 2nd call on infer()
    // Let's imagine everything is fine
    // OPENVINO_NOT_IMPLEMENTED;
}

bool AsyncInferRequest::wait_for(const std::chrono::milliseconds& /*timeout*/) {
    OPENVINO_NOT_IMPLEMENTED;
}

void AsyncInferRequest::cancel() {
    OPENVINO_NOT_IMPLEMENTED;
}

/*
void AsyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::ITensor> AsyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    // 3rd call on infer
    std::cout << "Asked for: " << port.get_any_name() << std::endl;
    // Cannot imagine everything is fine
    OPENVINO_NOT_IMPLEMENTED;
}
*/

std::vector<ov::SoPtr<ov::IVariableState>> AsyncInferRequest::query_state() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::ProfilingInfo> AsyncInferRequest::get_profiling_info() const {
    OPENVINO_NOT_IMPLEMENTED;
}

} // namespace mlir    
} // namespace ov
