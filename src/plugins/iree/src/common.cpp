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
        ss << std::exchange(sep, ", ")
           << "!torch.vtensor<"
           << io.get_partial_shape() << ","
           << io.get_element_type().get_type_name()
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

}
}
