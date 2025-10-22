#include "plugin/iree/common.hpp"

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

/*
iree_hal_element_types_t ov_to_iree_type(ov::element::Type_t t) {
  using ov::element::Type_t;
  switch (t) {
    case Type_t::boolean:
      return IREE_HAL_ELEMENT_TYPE_BOOL_8;

    case Type_t::i4:
      return IREE_HAL_ELEMENT_TYPE_SINT_4;
    case Type_t::i8:
      return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case Type_t::i16:
      return IREE_HAL_ELEMENT_TYPE_SINT_16;
    case Type_t::i32:
      return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case Type_t::i64:
      return IREE_HAL_ELEMENT_TYPE_SINT_64;

    case Type_t::u4:
      return IREE_HAL_ELEMENT_TYPE_UINT_4;
    case Type_t::u8:
      return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case Type_t::u16:
      return IREE_HAL_ELEMENT_TYPE_UINT_16;
    case Type_t::u32:
      return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case Type_t::u64:
      return IREE_HAL_ELEMENT_TYPE_UINT_64;

    case Type_t::bf16:
      return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    case Type_t::f16:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case Type_t::f32:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case Type_t::f64:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_64;

    case Type_t::f8e4m3:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN;
    case Type_t::f8e5m2:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2;
    case Type_t::f8e8m0:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E8M0_FNU;

    // Not representable in IREE (throw)
    case Type_t::dynamic:
    case Type_t::u1:
    case Type_t::u2:
    case Type_t::u3:
    case Type_t::u6:
    case Type_t::string:
    case Type_t::nf4:
    case Type_t::f4e2m1:
    default: {
      OPENVINO_THROW("Unsupported type ", type to string);
    }
  }
}
*/

}
}
