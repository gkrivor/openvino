#include "plugin/iree/compiled_model.hpp"
#include "plugin/iree/infer_request.hpp"
#include "plugin/iree/common.hpp"
#include "iree/compiler/embedding_api.h"
#include "iree/compiler/loader.h"

#include <fstream>

namespace ov {
namespace iree {

static std::mutex translators_mutex;
static std::map<std::string, std::string> ov_to_aten = {
    {"Abs", "torch.aten.abs"},
    {"Add", "torch.aten.add"},
    {"Multiply", "torch.aten.mul"},
    {"Subtract", "torch.aten.sub"},
    {"Divide", "torch.aten.div"},
    {"Power", "torch.aten.pow"},
    {"Minimum", "torch.aten.min"},
    {"Maximum", "torch.aten.max"},
    {"Equal", "torch.aten.eq"},
    {"NotEqual", "torch.aten.ne"},
    {"Greater", "torch.aten.gt"},
    {"GreaterEqual", "torch.aten.ge"},
    {"Less", "torch.aten.lt"},
    {"LessEqual", "torch.aten.le"},
    {"LogicalAnd", "torch.aten.logical_and"},
    {"LogicalOr", "torch.aten.logical_or"},
    {"LogicalXor", "torch.aten.logical_xor"},
    {"LogicalNot", "torch.aten.logical_not"},
    {"Negative", "torch.aten.neg"},
    {"Sign", "torch.aten.sign"},
    {"Floor", "torch.aten.floor"},
    {"Ceiling", "torch.aten.ceil"},
    {"Round", "torch.aten.round"},
    {"Trunc", "torch.aten.trunc"},
    {"Clamp", "torch.aten.clamp"},
    {"Sqrt", "torch.aten.sqrt"},
    {"Exp", "torch.aten.exp"},
    {"Log", "torch.aten.log"},
    {"Sin", "torch.aten.sin"},
    {"Cos", "torch.aten.cos"},
    {"Tan", "torch.aten.tan"},
    {"Asin", "torch.aten.asin"},
    {"Acos", "torch.aten.acos"},
    {"Atan", "torch.aten.atan"},
    {"Sinh", "torch.aten.sinh"},
    {"Cosh", "torch.aten.cosh"},
    {"Tanh", "torch.aten.tanh"},
    {"Asinh", "torch.aten.asinh"},
    {"Acosh", "torch.aten.acosh"},
    {"Atanh", "torch.aten.atanh"},
    {"Erf", "torch.aten.erf"},
    {"Relu", "torch.aten.relu"},
    {"Reshape", "torch.aten.view"}, // ???
    {"Broadcast", "torch.aten.broadcast_to"}, // ???
    // {"ShapeOf", "torch.aten._shape_as_tensor"},
    // {"Convert", "torch.aten._copy_to"},

};

std::map<std::string, translator_func>& get_translators() {
    static std::map<std::string, translator_func> translators_map;
    return translators_map;
}

std::map<std::string, translator_func>& get_post_processors() {
    static std::map<std::string, translator_func> post_processors_map;
    return post_processors_map;
}

bool register_translator(const std::string& op_type, translator_func func, bool is_post_processor) {
    std::lock_guard<std::mutex> lock(translators_mutex);
    if(is_post_processor == false) {
        get_translators()[op_type] = func;
    } else {
        std::cout << "Registering post processor for " << op_type << std::endl;
        get_post_processors()[op_type] = func;
    }
    return true;
}

void translator_ov_to_aten(const std::shared_ptr<const ov::Node>& node, std::ostream& ss, const std::string& torch_name) {
    ss << "    %" << getMLIRName(node)
       << " = \"" << torch_name << "\" ";

    genInputNames(ss, node);
    ss << " : ";
    genInputTypes(ss, node);
    genOutputTypes(ss, node);

    ss << "\n";
}

namespace lib {
void init_compiler() {
    static bool is_initialized = false;
    if(is_initialized) {
        return;
    }
    #ifdef _WIN32
    ireeCompilerLoadLibrary("IREECompiler.dll");
    #else
    ireeCompilerLoadLibrary("libIREECompiler.so");
    #endif
    ireeCompilerGlobalInitialize();
    // @todo: think how to call a GlobalShutdown...
    is_initialized = true;
}

void deinit_compiler() {
    ireeCompilerGlobalShutdown();
}

}

namespace {
void torch_variables(std::ostream& ss) {
    ss << "    %false = torch.constant.bool false\n"
       << "    %true = torch.constant.bool true\n"
       << "    %none = torch.constant.none\n";
    static std::vector<ov::element::Type> supported_types = {
        ov::element::u8,
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64,
        ov::element::f16,
        ov::element::f32,
        ov::element::f64,
        ov::element::bf16,
        ov::element::boolean
    };
    for(auto& t : supported_types) {
        ss << "    %dtype_" << ov_to_mlir_type(t) << " = torch.constant.int " << static_cast<int>(ov_to_torch_dtype(t)) << "\n";
    }
}
}

CompiledModel::CompiledModel(
    const std::shared_ptr<const ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::ICompiledModel(model, plugin),
    m_model(model), // @todo need to copy model
    m_compiled(nullptr),
    m_compiled_size(0) {
}

CompiledModel::~CompiledModel() {
    reset_compiled();
}

void CompiledModel::init() {
    m_module_name = m_model->get_friendly_name() + "." + m_model->get_friendly_name();

    std::shared_ptr<char> mlir_text = nullptr;
    size_t mlir_text_size = 0;
    {
        std::shared_ptr<std::stringstream> ss = std::make_shared<std::stringstream>();
        generate_mlir(*ss);
        // Store a string stream as a continuous buffer
        mlir_text_size = static_cast<size_t>(ss->tellp());
        mlir_text.reset(new char[mlir_text_size + 1], [](char* p) {delete[] p;});
        ss->seekg(0, std::ios::beg);
        ss->read(mlir_text.get(), mlir_text_size);
        *(mlir_text.get() + mlir_text_size) = 0;
#if 1
    std::fstream debug_dump("debug_dump.mlir", std::ios::binary | std::ios::out);
    debug_dump.write(reinterpret_cast<const char*>(mlir_text.get()), mlir_text_size);
    debug_dump.close();
#endif
    }

    iree_compiler_session_t *session = ireeCompilerSessionCreate();
    // @todo: need to be able customize it at this stage
    const char* iree_arg0 = "--iree-hal-target-device=local";
    const char* iree_arg1 = "--iree-hal-local-target-device-backends=llvm-cpu";
    const char* iree_arg2 = "--iree-llvmcpu-target-cpu=host";
    std::vector<const char*> iree_args { iree_arg0, iree_arg1, iree_arg2 };

    // @todo: Need to check each return status
    ireeCompilerSessionSetFlags(session, 3, iree_args.data());

    iree_compiler_source_t *source = NULL;
    ireeCompilerSourceWrapBuffer(session, "buffer", mlir_text.get(), mlir_text_size + 1, true, &source);

    // Use an invocation to compile from the input source to one or more outputs.
    iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(session);
    ireeCompilerInvocationParseSource(inv, source);
    ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD);

    // Output the compiled artifact to a file.
    iree_compiler_output_t *output = NULL;
    ireeCompilerOutputOpenMembuffer(&output);
    ireeCompilerInvocationOutputVMBytecode(inv, output);

    char* mlir_compiled = nullptr;
    // Getting a required buffer size
    ireeCompilerOutputMapMemory(output, reinterpret_cast<void**>(&mlir_compiled), &m_compiled_size);
    m_compiled = new uint8_t[m_compiled_size];
    memcpy_s(m_compiled, m_compiled_size, mlir_compiled, m_compiled_size);

    // Cleanup state.
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerOutputDestroy(output);
    ireeCompilerSourceDestroy(source);
    ireeCompilerSessionDestroy(session);

#if 1
    std::fstream debug_dump("debug_dump.vmfb", std::ios::binary | std::ios::out);
    debug_dump.write(reinterpret_cast<const char*>(m_compiled), m_compiled_size);
    debug_dump.close();
#endif
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    if(m_compiled == nullptr || m_compiled_size == 0) {
        OPENVINO_THROW("Model wasn't compiled");
    }
    auto internal_request = create_sync_infer_request();
    auto async_infer_request =
        std::make_shared<AsyncInferRequest>(std::static_pointer_cast<SyncInferRequest>(internal_request),
                                            get_task_executor(),
                                            get_callback_executor());
    return async_infer_request;
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<SyncInferRequest>(shared_from_this());
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    return m_model;
}

void CompiledModel::export_model(std::ostream& stream) const {
    generate_mlir(stream);
}

void CompiledModel::generate_mlir(std::ostream& stream) const {
    auto& ss = stream;

    ss << "module @" << m_model->get_friendly_name() << " {\n";

    ss << "  func.func @" << m_model->get_friendly_name() << "(";

    std::string delimeter = "";
    for(auto& arg: inputs()) {
        ss << delimeter << "%" << getMLIRName(arg) << ": !torch.vtensor<" << arg.get_partial_shape()
           << "," << ov_to_mlir_type(arg.get_element_type()) << ">";
        delimeter = ", ";
    }

    ss << ") -> ";

    delimeter = "";
    for(auto& res: outputs()) {
        ss << delimeter << "!torch.vtensor<" << res.get_partial_shape()
           << "," << ov_to_mlir_type(res.get_element_type()) << ">";
        delimeter = ", ";
    }

    ss << " {\n";

    // Insert variables for further usage later
    torch_variables(ss);

    auto& translators = get_translators();

    for (const auto& node : m_model->get_ordered_ops()) {
        const auto& type_info = node->get_type_info();
        // Direct translators from OV to ATen
        {
            // Find version-specific translator "Abs::opset1", and if it isn't found - common "Abs"
            auto it = ov_to_aten.find(std::string(type_info.name) + "::" + type_info.version_id);
            if(it == ov_to_aten.end()) {
                it = ov_to_aten.find(type_info.name);
            }
            if (it != ov_to_aten.end()) {
                // std::cout << "Found simple translator for " << type_info.name << " to " << it->second << std::endl;
                translator_ov_to_aten(node, ss, it->second);
                continue;
            }
        }
        // Custom translators from OV to ATen
        {
            // Find version-specific translator "Abs::opset1", and if it isn't found - common "Abs"
            auto it = translators.find(std::string(type_info.name) + "::" + type_info.version_id);
            if(it == translators.end()) {
                it = translators.find(type_info.name);
            }
            if (it != translators.end()) {
                auto fn = it->second;
                fn(node, ss);
                continue;
            }
        }
        // Failed to translate operation
        {
            ss << "  // Unsupported node: " << getMLIRName(node)
               << " (" << type_info.name << ")\n";
        }
    }
    delimeter = "    return ";
    for(auto& res: outputs()) {
        ss << delimeter << "%" << getMLIRName(res.get_node_shared_ptr()->input(0).get_source_output());
        delimeter = ", ";
    }
    ss << " : ";
    delimeter = "";
    for(auto& res: outputs()) {
        ss << delimeter << "!torch.vtensor<" << res.get_partial_shape()
           << "," << ov_to_mlir_type(res.get_element_type()) << ">";
        delimeter = ", ";
    }

    ss << "\n  }\n" // func.func
       << "}\n\n" // module
       << "{-#\n"
       << "dialect_resources: {\n"
       << "  builtin: {\n";
    auto& post_processors = get_post_processors();

    for (const auto& node : m_model->get_ordered_ops()) {
        const auto& type_info = node->get_type_info();
        {
            // Find version-specific post processor "Abs::opset1", and if it isn't found - common "Abs"
            auto it = post_processors.find(std::string(type_info.name) + "::" + type_info.version_id);
            if(it == post_processors.end()) {
                it = post_processors.find(type_info.name);
            }
            if (it != post_processors.end()) {
                auto fn = it->second;
                fn(node, ss);
                continue;
            }
        }
    }
    ss << "  }\n" // builtin
       << "}\n" // dialect_resources
       << "#-}\n";
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    (void)name;
    OPENVINO_THROW("Not implemented");
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    (void)properties;
    OPENVINO_THROW("Not implemented");
}

void CompiledModel::reset_compiled() {
    if(m_compiled) {
        delete[] m_compiled;
    }
    m_compiled = nullptr;
    m_compiled_size = 0;
}

} // namespace iree
} // namespace ov
