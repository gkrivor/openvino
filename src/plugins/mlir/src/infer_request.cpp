#include "plugin/mlir/infer_request.hpp"
#include "plugin/mlir/compiled_model.hpp"
#pragma warning(push)
#pragma warning(disable : 4146)
#pragma warning(disable : 4200)
#include "iree/runtime/api.h"
#pragma warning(pop)
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace iree {

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
    , m_iree_session(nullptr)
    , m_compiled_model(infer_request->get_compiled_model()) {
}

AsyncInferRequest::~AsyncInferRequest() {
    if(m_iree_device) {
        iree_hal_device_release(m_iree_device);
    }
    m_iree_device = nullptr;
}

void AsyncInferRequest::infer() {
    iree_status_t status = nullptr;
    std::cout << "Device creation...";
    status = iree_runtime_instance_try_create_default_device(iree::instance, iree_make_cstring_view("local-task"), &m_iree_device);
    if(status) {
        OPENVINO_THROW("Error creating iree device");
    }
    std::cout << "OK\n";

    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    std::cout << "Session creation...";
    status = iree_runtime_session_create_with_device(
      iree::instance, &session_options, m_iree_device,
      iree_runtime_instance_host_allocator(iree::instance), &m_iree_session);
    std::cout << "OK\n";
    if(status) {
        OPENVINO_THROW("Error creating iree session");
    }

    auto compiled_model = std::dynamic_pointer_cast<const CompiledModel>(m_compiled_model);
    auto compiled_mlir = compiled_model->get_compiled_mlir();
    size_t compiled_mlir_size = compiled_model->get_compiled_mlir_size();
    std::cout << "Module creation...";
    status = iree_runtime_session_append_bytecode_module_from_memory(
        m_iree_session, iree_make_const_byte_span(compiled_mlir, compiled_mlir_size),
        iree_allocator_null());
    if(status) {
        OPENVINO_THROW("Error creating iree model");
    }
    std::cout << "OK\n";

    iree_runtime_call_t call;
    std::cout << "Initialize by name...";
    status = iree_runtime_call_initialize_by_name(
        m_iree_session, iree_make_cstring_view(compiled_model->get_module_name().c_str()), &call);
    if(status) {
        OPENVINO_THROW("Error initialization module");
    }
    std::cout << "OK\n";
    iree_hal_allocator_t* device_allocator =
        iree_runtime_session_device_allocator(m_iree_session);
    iree_allocator_t host_allocator =
        iree_runtime_session_host_allocator(m_iree_session);

    iree_hal_buffer_params_t buffer_params = {
                // Intended usage of the buffer (transfers, dispatches, etc):
                IREE_HAL_BUFFER_USAGE_DEFAULT,
                // Access to allow to this memory:
                IREE_HAL_MEMORY_ACCESS_ALL,
                // Where to allocate (host or device):
                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                IREE_HAL_QUEUE_AFFINITY_ANY,
                0
            };

    for(auto node: get_inputs()) {
        iree_hal_buffer_view_t* buffer = NULL;
        //static const iree_hal_dim_t arg0_shape[1] = {4};
        //static const float arg0_data[4] = {1.0f, 1.1f, 1.2f, 1.3f};
        std::cout << "Creating input " << node.get_any_name() << "...";
        std::vector<iree_hal_dim_t> arg_shape;
        for(auto dim: node.get_shape()) {
            arg_shape.push_back(dim);
        }
        auto tensor = m_sync_request->get_tensor(node);
        status = iree_hal_buffer_view_allocate_buffer_copy(
            m_iree_device, device_allocator,
            // Shape rank and dimensions:
            arg_shape.size(), arg_shape.data(),
            // Element type:
            IREE_HAL_ELEMENT_TYPE_FLOAT_32,
            // Encoding type:
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
            buffer_params,
            // The actual heap buffer to wrap or clone and its allocator:
            iree_make_const_byte_span(tensor->data(), tensor->get_byte_size()),
            // Buffer view + storage are returned and owned by the caller:
            &buffer);
        if(status) {
            OPENVINO_THROW("Error creating input");
        }
        std::cout << "OK\n";
        iree_hal_buffer_view_fprint(
          stdout, buffer, /*max_element_count=*/4096, host_allocator);
        std::cout << std::endl;
        std::cout << "Pushing input to call...";
        // Add to the call inputs list (which retains the buffer view).
        status = iree_runtime_call_inputs_push_back_buffer_view(&call, buffer);
        if(status) {
            OPENVINO_THROW("Error pushing input");
        }
        std::cout << "OK\n";
        // Since the call retains the buffer view we can release it here.
        iree_hal_buffer_view_release(buffer);
    }

    std::cout << "Invoking runtime...";
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
    if(status) {
        OPENVINO_THROW("Error invoking runtime");
    }
    std::cout << "OK\n";

    for(auto node : get_outputs()) {
        std::cout << "Reading output " << node.get_any_name() << "...";
        iree_hal_buffer_view_t* buffer = NULL;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &buffer);
        if(status) {
            OPENVINO_THROW("Cannot get output buffer view");
        }
        std::cout << "OK\n";
        iree_hal_buffer_view_fprint(
          stdout, buffer, /*max_element_count=*/4096, host_allocator);
        std::cout << std::endl;
        auto tensor = m_sync_request->get_tensor(node);
        iree_hal_buffer_mapping_t buffer_mapping = {{0}};
        std::cout << "Mapping buffer...";
        status = iree_hal_buffer_map_range(
            iree_hal_buffer_view_buffer(buffer), IREE_HAL_MAPPING_MODE_SCOPED,
            IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &buffer_mapping);
        if(status) {
            OPENVINO_THROW("Cannot map buffer");
        }
        std::cout << "OK\n";
        //memcpy_s(tensor->data(), tensor->get_byte_size(), iree_hal_buffer_view_buffer(buffer), iree_hal_buffer_view_byte_length(buffer));
        memcpy_s(tensor->data(), tensor->get_byte_size(), buffer_mapping.contents.data, buffer_mapping.contents.data_length);
        iree_hal_buffer_unmap_range(&buffer_mapping);
        iree_hal_buffer_view_release(buffer);
    }
    iree_runtime_call_deinitialize(&call);
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

} // namespace iree    
} // namespace ov
