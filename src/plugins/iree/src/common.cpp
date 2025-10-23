#include "plugin/iree/common.hpp"
// For types convertion
#pragma warning(push)
#pragma warning(disable : 4146)
#pragma warning(disable : 4200)
#include "iree/hal/buffer_view.h"
#pragma warning(pop)
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace iree {

namespace {
template <typename Range>
void genVtensorTypeList(std::ostream& ss, const Range& range) {
    std::string sep;
    for (const auto& io : range) {
        std::string type_name = ov_to_mlir_type(io.get_element_type());
        ss << std::exchange(sep, ", ")
           << "!torch.vtensor<"
           << io.get_partial_shape() << ","
           << type_name
           << ">";
    }
}
} // namespace

void genInputNames(std::ostream& ss, const std::shared_ptr<const ov::Node>& node) {
    ss << "(";
    std::string sep;
    for (const auto& io : node->inputs()) {
        ss << std::exchange(sep, ", ")
           << "%" << io.get_source_output().get_node()->get_friendly_name();
    }
    ss << ")";
}

void genInputTypes(std::ostream& ss, const std::shared_ptr<const ov::Node>& node) {
    ss << "(";
    genVtensorTypeList(ss, node->inputs());
    ss << ")";
}

void genOutputTypes(std::ostream& ss, const std::shared_ptr<const ov::Node>& node) {
    // Insert " -> " only if inputs size > 0
    if (!node->inputs().empty()) {
        ss << " -> ";
    }
    genVtensorTypeList(ss, node->outputs());
}

/// @brief Function converts OpenVINO's element type to IREE's type. Function declaration is
/// absent in header and MUST be declared manually in *.cpp file it is using.
/// It is done because simple type conversion requires comparable huge headers to include.
/// @param t Type for conversion
/// @return Returns converted type or throws and exception if conversion is failed
iree_hal_element_types_t ov_to_iree_type(ov::element::Type t) {
  switch (t) {
    case ov::element::boolean:
      return IREE_HAL_ELEMENT_TYPE_BOOL_8;

    case ov::element::i4:
      return IREE_HAL_ELEMENT_TYPE_SINT_4;
    case ov::element::i8:
      return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case ov::element::i16:
      return IREE_HAL_ELEMENT_TYPE_SINT_16;
    case ov::element::i32:
      return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case ov::element::i64:
      return IREE_HAL_ELEMENT_TYPE_SINT_64;

    case ov::element::u4:
      return IREE_HAL_ELEMENT_TYPE_UINT_4;
    case ov::element::u8:
      return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case ov::element::u16:
      return IREE_HAL_ELEMENT_TYPE_UINT_16;
    case ov::element::u32:
      return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case ov::element::u64:
      return IREE_HAL_ELEMENT_TYPE_UINT_64;

    case ov::element::bf16:
      return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    case ov::element::f16:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case ov::element::f32:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case ov::element::f64:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_64;

    case ov::element::f8e4m3:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN;
    case ov::element::f8e5m2:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2;
    case ov::element::f8e8m0:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E8M0_FNU;

    // Not representable in IREE (throw)
    case ov::element::dynamic:
    case ov::element::u1:
    case ov::element::u2:
    case ov::element::u3:
    case ov::element::u6:
    case ov::element::string:
    case ov::element::nf4:
    case ov::element::f4e2m1:
    default: {
      OPENVINO_THROW("Unsupported type ", t.get_type_name());
    }
  }
}

/// @brief Function converts OpenVINO's element type to MLIR's type.
/// Reference for types:
/// https://github.com/llvm/torch-mlir/blob/4e7f20444e0163e65c837b20841205b21a3f8561/include/torch-mlir/Dialect/Torch/IR/TorchTypes.td#L117C1-L135C47
/// @param t Type for conversion
/// @return Returns converted type name or throws and exception if conversion is failed
std::string ov_to_mlir_type(ov::element::Type t) {
  switch (t) {
    case ov::element::boolean:
      return "i1";

    case ov::element::i8:
      return "si8";
    case ov::element::i16:
      return "si16";
    case ov::element::i32:
      return "si32";
    case ov::element::i64:
      return "si64";

    case ov::element::u8:
      return "ui8";

    case ov::element::bf16:
      return "bf16";
    case ov::element::f16:
      return "f16";
    case ov::element::f32:
      return "f32";
    case ov::element::f64:
      return "f64";

    // Not representable in MLIR (throw)
    case ov::element::dynamic:
    case ov::element::u1:
    case ov::element::u2:
    case ov::element::u3:
    case ov::element::u4:
    case ov::element::u6:
    case ov::element::u16:
    case ov::element::u32:
    case ov::element::u64:
    case ov::element::i4:
    case ov::element::string:
    case ov::element::nf4:
    case ov::element::f4e2m1:
    case ov::element::f8e4m3:
    case ov::element::f8e5m2:
    case ov::element::f8e8m0:
    default: {
      OPENVINO_THROW("Unsupported type ", t.get_type_name());
    }
  }
}

/// @brief Function converts OpenVINO's element type to Torch's dtype.
/// Reference for types:
/// https://github.com/llvm/torch-mlir/blob/4e7f20444e0163e65c837b20841205b21a3f8561/include/torch-mlir/Dialect/Torch/Utils/TorchUpstream.h#L88C1-L116C52
/// @param t Type for conversion
/// @return Returns converted type id or throws and exception if conversion is failed
uint8_t ov_to_torch_dtype(ov::element::Type t) {
  switch (t) {
    case ov::element::boolean:
      return 11;

    case ov::element::i8:
      return 1;
    case ov::element::i16:
      return 2;
    case ov::element::i32:
      return 3;
    case ov::element::i64:
      return 4;

    case ov::element::u8:
      return 0;

    case ov::element::bf16:
      return 15;
    case ov::element::f16:
      return 5;
    case ov::element::f32:
      return 6;
    case ov::element::f64:
      return 7;

    // Not representable in Torch's dtype (throw)
    case ov::element::dynamic:
    case ov::element::u1:
    case ov::element::u2:
    case ov::element::u3:
    case ov::element::u4:
    case ov::element::u6:
    case ov::element::u16:
    case ov::element::u32:
    case ov::element::u64:
    case ov::element::i4:
    case ov::element::string:
    case ov::element::nf4:
    case ov::element::f4e2m1:
    case ov::element::f8e4m3:
    case ov::element::f8e5m2:
    case ov::element::f8e8m0:
    default: {
      OPENVINO_THROW("Unsupported type ", t.get_type_name());
    }
  }
}

}
}
