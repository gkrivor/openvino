#include "plugin/mlir/common.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"

using namespace ov::mlir;

#define MAX_INLINE_CONST 20

namespace {
    void translate_scalar_constant(const std::shared_ptr<const ov::op::v0::Constant>& constant, std::ostream& ss) {
        ss << " %" << constant->get_friendly_name()
        << " = torch.constant.";
        
        std::string const_type;
        switch (constant->get_element_type())
        {
        case ov::element::f64:
        case ov::element::f32:
        case ov::element::bf16:
        case ov::element::f16:
            const_type = "float"; break;
        case ov::element::i64:
        case ov::element::i32:
        case ov::element::i16:
        case ov::element::i8:
        case ov::element::i4:
        case ov::element::u64:
        case ov::element::u32:
        case ov::element::u16:
        case ov::element::u8:
        case ov::element::u4:
            const_type = "int"; break;
        case ov::element::boolean:
            const_type = "bool"; break;
        default:
            OPENVINO_THROW("Constant type isn't supported");
        }
        ss << const_type << " ";

        switch (constant->get_element_type())
        {
        case ov::element::f64:
            ss << constant->cast_vector<double>()[0];
            break;
        case ov::element::f32:
            ss << constant->cast_vector<float>()[0];
            break;
        case ov::element::bf16:
            ss << constant->cast_vector<ov::bfloat16>()[0];
            break;
        case ov::element::f16:
            ss << constant->cast_vector<ov::float16>()[0];
            break;
        case ov::element::i64:
            ss << constant->cast_vector<int64_t>()[0];
            break;
        case ov::element::i32:
            ss << constant->cast_vector<int32_t>()[0];
            break;
        case ov::element::i16:
            ss << constant->cast_vector<int16_t>()[0];
            break;
        case ov::element::i8:
            ss << constant->cast_vector<int8_t>()[0];
            break;
        case ov::element::i4:
            ss << ((constant->cast_vector<int8_t>()[0] & 0xF0) >> 4);
            break;
        case ov::element::u64:
            ss << constant->cast_vector<uint64_t>()[0];
            break;
        case ov::element::u32:
            ss << constant->cast_vector<uint32_t>()[0];
            break;
        case ov::element::u16:
            ss << constant->cast_vector<uint16_t>()[0];
            break;
        case ov::element::u8:
            ss << constant->cast_vector<uint8_t>()[0];
            break;
        case ov::element::u4:
            ss << ((constant->cast_vector<uint8_t>()[0] & 0xF0) >> 4);
            break;
        case ov::element::boolean:
            ss << (constant->cast_vector<uint8_t>()[0] != 0);
            break;
        }
    }
/*
Example of translation:

module @test_abs {
  func.func @test_abs() -> !torch.vtensor<[2],f16> {
    %x = torch.vtensor.literal(dense<[-123.123, 15.20]> : tensor<[2]xf16>) : !torch.vtensor<[2],f16>
    %y = "torch.aten.abs" (%x) : (!torch.vtensor<[2],f16>) -> !torch.vtensor<[2],f16>
  return %y : !torch.vtensor<[2],f16>
  }
}
*/
    template<typename DataType>
    void print_values(const std::vector<DataType>& data, const ov::Shape& shape, std::ostream& ss) {
        std::string sep;
        for(size_t i = 0; i < data.size(); ++i) {
            size_t accum = 1;
            for(auto dim = shape.rbegin(); dim != shape.rend(); ++dim) {
                accum *= *dim;
                if (((i % accum) == 0) && ((i / accum) > 0)) {
                    ss << "]";
                }
            }
            accum = 1;
            sep = ", ";
            for(auto dim = shape.rbegin(); dim != shape.rend(); ++dim) {
                accum *= *dim;
                if ((i % accum) == 0) {
                    if((i / accum) > 0) {
                        ss << sep;
                    }
                    ss << "[";
                    sep = "";
                }
            }
            ss << sep << data[i];
            sep = ", ";
        }
        for(size_t i = 0; i < shape.size(); ++i) {
            ss << "]";
        }
    }

    void translate_inline_constant(const std::shared_ptr<const ov::op::v0::Constant>& constant, std::ostream& ss) {
        ss << " %" << constant->get_friendly_name()
        << " = torch.vtensor.literal(dense<";
    
        const auto& const_shape = constant->get_output_partial_shape(0).to_shape();

        switch (constant->get_element_type())
        {
        case ov::element::f64:
            print_values(constant->cast_vector<double>(), const_shape, ss);
            break;
        case ov::element::f32:
            print_values(constant->cast_vector<float>(), const_shape, ss);
            break;
        case ov::element::bf16:
            print_values(constant->cast_vector<ov::bfloat16>(), const_shape, ss);
            break;
        case ov::element::f16:
            print_values(constant->cast_vector<ov::float16>(), const_shape, ss);
            break;
        case ov::element::i64:
            print_values(constant->cast_vector<int64_t>(), const_shape, ss);
            break;
        case ov::element::i32:
            print_values(constant->cast_vector<int32_t>(), const_shape, ss);
            break;
        case ov::element::i16:
            print_values(constant->cast_vector<int16_t>(), const_shape, ss);
            break;
        case ov::element::i8:
            print_values(constant->cast_vector<int8_t>(), const_shape, ss);
            break;
        case ov::element::u64:
            print_values(constant->cast_vector<uint64_t>(), const_shape, ss);
            break;
        case ov::element::u32:
            print_values(constant->cast_vector<uint32_t>(), const_shape, ss);
            break;
        case ov::element::u16:
            print_values(constant->cast_vector<uint16_t>(), const_shape, ss);
            break;
        case ov::element::u8:
            print_values(constant->cast_vector<uint8_t>(), const_shape, ss);
            break;
        default:
            OPENVINO_THROW("Constant type isn't supported");
        }
        ss << "> : tensor<";

        std::string sep;
        for (auto& d : const_shape) {
            ss << sep << d;
            sep = "x";
        }
        
        ss << "x" << constant->get_output_element_type(0).get_type_name() << ">) : "
           << "!torch.vtensor<" << constant->get_output_partial_shape(0)
           << "," << constant->get_output_element_type(0).get_type_name() << ">";
    }

    void translate_resource_constant(const std::shared_ptr<const ov::op::v0::Constant>& constant, std::ostream& ss) {
        const auto const_name = constant->get_friendly_name();
        ss << " %" << const_name << " = torch.vtensor.literal(dense_resource<"
           << const_name << "_res> : tensor<";

        const auto& const_shape = constant->get_output_partial_shape(0).to_shape();
        std::string sep;
        for (auto& d : const_shape) {
            ss << sep << d;
            sep = "x";
        }
        
        ss << "x" << constant->get_output_element_type(0).get_type_name() << ">) : "
           << "!torch.vtensor<" << constant->get_output_partial_shape(0)
           << "," << constant->get_output_element_type(0).get_type_name() << ">";
    }
}

static void translator_constant(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
    auto constant = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node);
    if (!constant) return;

    const auto& const_shape = constant->get_output_partial_shape(0).to_shape();
    const auto const_shape_size = ov::shape_size(const_shape);

    if(const_shape_size == 0) {
        translate_scalar_constant(constant, ss);
    } else if(const_shape_size <= MAX_INLINE_CONST) {
        translate_inline_constant(constant, ss);
    } else {
        const auto& rt_info = constant->get_rt_info();
        if(rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) != rt_info.end()) {
            // External file
        } else {
            translate_resource_constant(constant, ss);
        }
    }

/*
    genInputNames(ss, node);
    ss << " : ";
    genInputTypes(ss, node);
    genOutputTypes(ss, node);
*/
    ss << "\n";
}

EMIT_REG("Constant", translator_constant);
