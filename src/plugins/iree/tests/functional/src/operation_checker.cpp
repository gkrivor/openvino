// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation_checker.hpp"

#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/all_close_f.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

namespace {
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, testing::AssertionResult>::type
compare_values(const ov::Tensor& expected, const ov::Tensor& result, const size_t tolerance_bits) {
    return ov::test::utils::all_close_f(expected, result, static_cast<int>(tolerance_bits));
}

testing::AssertionResult compare_with_tolerance(const std::vector<float>& expected,
                                                const std::vector<float>& results,
                                                const float tolerance) {
    auto comparison_result = testing::AssertionSuccess();

    std::stringstream msg;
    msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

    bool rc = true;

    for (std::size_t j = 0; j < expected.size(); ++j) {
        float diff = std::fabs(results[j] - expected[j]);
        if (diff > tolerance) {
            msg << expected[j] << " is not close to " << results[j] << " at index " << j << "\n";
            rc = false;
        }
    }

    if (!rc) {
        comparison_result = testing::AssertionFailure();
        comparison_result << msg.str();
    }

    return comparison_result;
}

testing::AssertionResult compare_with_fp_tolerance(const ov::Tensor& expected_tensor,
                                                   const ov::Tensor& result_tensor,
                                                   const float tolerance) {
    OPENVINO_ASSERT(expected_tensor.get_element_type() == ov::element::f32);

    std::vector<float> expected(expected_tensor.get_size());
    ov::Tensor expected_view(expected_tensor.get_element_type(), expected_tensor.get_shape(), expected.data());
    expected_tensor.copy_to(expected_view);

    std::vector<float> result(result_tensor.get_size());
    ov::Tensor result_view(result_tensor.get_element_type(), result_tensor.get_shape(), result.data());
    result_tensor.copy_to(result_view);

    return compare_with_tolerance(expected, result, tolerance);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, testing::AssertionResult>::type
compare_values(const ov::Tensor& expected, const ov::Tensor& result, const size_t) {
    return ov::test::utils::all_close(expected, result);
}

// used for float16 and bfloat 16 comparisons
template <typename T>
typename std::enable_if<std::is_class<T>::value, testing::AssertionResult>::type
compare_values(const ov::Tensor& expected_tensor, const ov::Tensor& result_tensor, const size_t tolerance_bits) {
    auto expected_tensor_converted =
        ov::test::utils::make_tensor_with_precision_convert(expected_tensor, ov::element::f64);
    auto result_tensor_converted = ov::test::utils::make_tensor_with_precision_convert(result_tensor, ov::element::f64);

    return ov::test::utils::all_close_f(expected_tensor_converted,
                                        result_tensor_converted,
                                        static_cast<int>(tolerance_bits));
}
};  // namespace

namespace ov {
namespace iree {

std::pair<testing::AssertionResult, size_t> OperationChecker::compare_results(size_t tolerance_bits) {
    auto res = testing::AssertionSuccess();
    size_t output_idx = 0;
    for (; output_idx < m_outputs_count; ++output_idx) {
        const auto& result_tensor = m_request->get_output_tensor(output_idx);
        const auto& exp_result = m_ref_request->get_output_tensor(output_idx);

        const auto& element_type = result_tensor.get_element_type();
        const auto& res_shape = result_tensor.get_shape();
        const auto& exp_shape = exp_result.get_shape();

        if (exp_shape != res_shape) {
            res = testing::AssertionFailure();
            res << "Computed data shape(" << res_shape << ") does not match the expected shape(" << exp_shape
                << ") for output " << output_idx << std::endl;
            break;
        }

        switch (element_type) {
        case ov::element::Type_t::f8e5m2:
            res = compare_values<ov::float8_e5m2>(exp_result, result_tensor, tolerance_bits);
            break;
        case ov::element::Type_t::f8e4m3:
            res = compare_values<ov::float8_e4m3>(exp_result, result_tensor, tolerance_bits);
            break;
        case ov::element::Type_t::f16:
            res = compare_values<ov::float16>(exp_result, result_tensor, tolerance_bits);
            break;
        case ov::element::Type_t::bf16:
            res = compare_values<ov::bfloat16>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::f32:
            res = compare_values<float>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::f64:
            res = compare_values<double>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i4:
            res = compare_values<uint8_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i8:
            res = compare_values<int8_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i16:
            res = compare_values<int16_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i32:
            res = compare_values<int32_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i64:
            res = compare_values<int64_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u4:
            res = compare_values<uint8_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u8:
            res = compare_values<uint8_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u16:
            res = compare_values<uint16_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u32:
            res = compare_values<uint32_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u64:
            res = compare_values<uint64_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::boolean:
            res = compare_values<char>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::string: {
            res = ::testing::AssertionSuccess();
            const std::string* exp_strings = exp_result.data<std::string>();
            const std::string* res_strings = result_tensor.data<std::string>();
            for (size_t i = 0; i < exp_result.get_size(); ++i) {
                if (exp_strings[i] != res_strings[i]) {
                    res = ::testing::AssertionFailure() << "Wrong string value at index " << i << ", expected \""
                                                        << exp_strings[i] << "\" got \"" << res_strings[i] << "\"";
                    break;
                }
            }
        } break;
        default:
            res = testing::AssertionFailure() << "Unsupported data type encountered in 'res' method";
        }
        if (res == testing::AssertionFailure())
            break;
    }
    return std::make_pair(res, output_idx);
}

testing::AssertionResult OperationChecker::compare_results_with_tolerance_as_fp(float tolerance) {
    auto comparison_result = testing::AssertionSuccess();

    for (size_t i = 0; i < m_outputs_count; ++i) {
        const auto& result_tensor = m_request->get_output_tensor(i);
        const auto& exp_result = m_ref_request->get_output_tensor(i);
        const auto& element_type = result_tensor.get_element_type();

        const auto& expected_shape = exp_result.get_shape();
        const auto& result_shape = result_tensor.get_shape();

        if (expected_shape != result_shape) {
            comparison_result = testing::AssertionFailure();
            comparison_result << "Computed data shape(" << result_shape << ") does not match the expected shape("
                              << expected_shape << ") for output " << i << std::endl;
            break;
        }

        switch (element_type) {
        case element::Type_t::f32:
            comparison_result = compare_with_fp_tolerance(exp_result, result_tensor, tolerance);
            break;
        case element::Type_t::i32:
            comparison_result = compare_values<int32_t>(exp_result, result_tensor, 0);
            break;
        default:
            comparison_result = testing::AssertionFailure() << "Unsupported data type encountered in "
                                                               "'compare_results_with_tolerance_as_fp' method";
        }

        if (comparison_result == testing::AssertionFailure()) {
            break;
        }
    }

    return comparison_result;
}

OperationChecker::OperationChecker(const std::string& dev) : m_model(nullptr), m_dev(dev) {
    m_core = std::make_shared<ov::Core>();
    try {
        // Register IREE plugin
        m_core->register_plugin(
            ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                               std::string("openvino_iree_plugin") + OV_BUILD_POSTFIX),
            "IREE");
    } catch (...) {
    }
}

void OperationChecker::set_model(const std::shared_ptr<ov::Model>& model) {
    m_model = model;
    AnyMap config = {{ov::hint::inference_precision.name(), ov::element::f32}};
    m_reference_model = std::make_shared<ov::CompiledModel>(m_core->compile_model(m_model, m_dev, config));
    m_compiled_model = std::make_shared<ov::CompiledModel>(m_core->compile_model(m_model, "IREE"));
    m_inputs_count = m_reference_model->inputs().size();
    m_outputs_count = m_reference_model->outputs().size();
    EXPECT_EQ(m_compiled_model->inputs().size(),  m_inputs_count);
    EXPECT_EQ(m_compiled_model->outputs().size(), m_outputs_count);
    m_ref_request = std::make_shared<ov::InferRequest>(m_reference_model->create_infer_request());
    m_request = std::make_shared<ov::InferRequest>(m_compiled_model->create_infer_request());
}


void OperationChecker::run(const size_t tolerance_bits) {
    m_ref_request->infer();
    m_request->infer();

    const auto res = compare_results(tolerance_bits);

    if (res.first != testing::AssertionSuccess()) {
        std::cout << "Results comparison failed for output: " << res.second << std::endl;
        std::cout << res.first.message() << std::endl;
    }

    m_input_index = 0;

    EXPECT_TRUE(res.first);
}

void OperationChecker::run_with_tolerance_as_fp(const float tolerance) {
    m_ref_request->infer();
    m_request->infer();

    const auto res = compare_results_with_tolerance_as_fp(tolerance);

    if (res != testing::AssertionSuccess()) {
        std::cout << res.message() << std::endl;
    }

    m_input_index = 0;

    EXPECT_TRUE(res);
}

}  // namespace test
}  // namespace ov
