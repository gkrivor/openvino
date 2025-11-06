#pragma once

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_iterator.hpp"

namespace ov {
class Core;
class InferRequest;
class Model;
class Shape;
class Tensor;
class CompiledModel;

namespace iree {
class OperationChecker {
    std::shared_ptr<ov::Model> m_model;
public:
    OperationChecker(const std::string& dev = "CPU");

    void set_model(const std::shared_ptr<ov::Model>& model);

    /// \brief This method is used to eliminate issue caused by calling .data for vector<bool>
    template <typename T, typename std::enable_if<!std::is_same<T, bool>::value, bool>::type = true>
    void copy_values_to_tensor(ov::Tensor& tensor, const std::vector<T>& values) {
        if (!ov::element::is_nibble_type(tensor.get_element_type())) {
            std::copy(values.begin(), values.end(), tensor.data<T>());
        } else {
            size_t size = sizeof(T) * values.size();
            if (tensor.get_byte_size() < size)
                size = tensor.get_byte_size();
            std::memcpy(tensor.data(), values.data(), size);
        }
    }
    template <typename T, typename std::enable_if<std::is_same<T, bool>::value, bool>::type = true>
    void copy_values_to_tensor(ov::Tensor& tensor, const std::vector<bool>& values) {
        std::copy(values.begin(), values.end(), tensor.data<bool>());
    }

    template <typename T>
    void add_input(const ov::Shape& shape, const std::vector<T>& values) {
        const auto params = m_model->get_parameters();
        OPENVINO_ASSERT(m_input_index < params.size(), "All function parameters already have inputs.");

        const auto& input_pshape = params.at(m_input_index)->get_partial_shape();
        OPENVINO_ASSERT(input_pshape.compatible(shape),
                        "Provided input shape ",
                        shape,
                        " is not compatible with OpenVINO model's expected input shape ",
                        input_pshape,
                        " for input ",
                        m_input_index);

        auto t_shape = m_request->get_input_tensor(m_input_index).get_shape();
        bool is_dynamic = false;
        for (const auto& dim : t_shape) {
            if (!dim) {
                is_dynamic = true;
                break;
            }
        }

        if (is_dynamic) {
            {
                ov::Tensor tensor(params.at(m_input_index)->get_element_type(), shape);
                copy_values_to_tensor<T>(tensor, values);
                m_ref_request->set_input_tensor(m_input_index, tensor);
            }
            {
                ov::Tensor tensor(params.at(m_input_index)->get_element_type(), shape);
                copy_values_to_tensor<T>(tensor, values);
                m_request->set_input_tensor(m_input_index, tensor);
            }
        } else {
            {
                auto tensor = m_ref_request->get_input_tensor(m_input_index);
                OPENVINO_ASSERT(tensor.get_size() >= values.size(),
                                "Tensor and values have different sizes. Tensor (",
                                tensor.get_shape(),
                                ") size: ",
                                tensor.get_size(),
                                " and values size is ",
                                values.size());
                copy_values_to_tensor<T>(tensor, values);
            }
            {
                auto tensor = m_request->get_input_tensor(m_input_index);
                OPENVINO_ASSERT(tensor.get_size() >= values.size(),
                                "Tensor and values have different sizes. Tensor (",
                                tensor.get_shape(),
                                ") size: ",
                                tensor.get_size(),
                                " and values size is ",
                                values.size());
                copy_values_to_tensor<T>(tensor, values);
            }
        }

        ++m_input_index;
    }

    template <typename T>
    void add_input(const std::vector<T>& values) {
        const auto& input_pshape = m_model->get_parameters().at(m_input_index)->get_partial_shape();

        OPENVINO_ASSERT(input_pshape.is_static(),
                        "Input number ",
                        m_input_index,
                        " in the tested graph has dynamic shape. You need to provide ",
                        "shape information when setting values for this input.");

        add_input<T>(input_pshape.to_shape(), values);
    }

    template <typename T>
    void add_multiple_inputs(const std::vector<std::vector<T>>& vector_of_values) {
        for (const auto& value : vector_of_values) {
            add_input<T>(value);
        }
    }

    template <typename T>
    void add_input_from_file(const ov::Shape& shape, const std::string& basepath, const std::string& filename) {
        const auto filepath = ov::util::path_join({basepath, filename});
        add_input_from_file<T>(shape, filepath);
    }

    template <typename T>
    void add_input_from_file(const std::string& basepath, const std::string& filename) {
        const auto filepath = ov::util::path_join({basepath, filename});
        add_input_from_file<T>(filepath);
    }

    template <typename T>
    void add_input_from_file(const ov::Shape& shape, const std::string& filepath) {
        const auto value = read_binary_file<T>(filepath);
        add_input<T>(shape, value);
    }

    template <typename T>
    void add_input_from_file(const std::string& filepath) {
        const auto value = read_binary_file<T>(filepath);
        add_input<T>(value);
    }

    void run(const size_t tolerance_bits = 5);

    void run_with_tolerance_as_fp(const float tolerance = 1.0e-5f);

private:
    std::string m_dev;
    std::shared_ptr<ov::Core> m_core;
    std::shared_ptr<ov::CompiledModel> m_compiled_model, m_reference_model;
    std::shared_ptr<ov::InferRequest> m_request, m_ref_request;
    size_t m_input_index = 0;
    size_t m_inputs_count = 0;
    size_t m_outputs_count = 0;
    std::pair<testing::AssertionResult, size_t> compare_results(size_t tolerance_bits);
    testing::AssertionResult compare_results_with_tolerance_as_fp(float tolerance_bits);
};
}
}
