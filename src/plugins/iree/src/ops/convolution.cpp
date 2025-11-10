#include "plugin/iree/common.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/core/except.hpp"

using namespace ov::iree;

static void create_int_list(const std::string& node_name, const std::string list_name, const std::vector<size_t>& list, std::ostream& ss) {
    ss << "    " << node_name << "_" << list_name << " = torch.prim.ListConstruct ";
    std::string delimiter = "";
    for(auto element : list) {
        ss << delimiter;
        switch(element) {
            case 0: ss << "%const_zero"; break;
            case 1: ss << "%const_one"; break;
            default: ss << node_name << "_int_" << std::to_string(element); break;
        }
        delimiter = ", ";
    }
    ss << " : (";
    delimiter = "";
    for(size_t i = 0; i < list.size(); ++i) {
        ss << delimiter << "!torch.int";
        delimiter = ", ";
    }
    ss << ") -> !torch.list<int>\n";
}

static void translator_convolution_1d(const std::shared_ptr<const ov::op::v1::Convolution>& conv, std::ostream& ss) {
    const auto node = std::dynamic_pointer_cast<const ov::Node>(conv);
    auto& input0 = conv->input_value(0);
    auto& input1 = conv->input_value(1);
    auto node_name = std::string("%") + getMLIRName(node);
    auto input0_name = getMLIRName(input0);
    auto input1_name = getMLIRName(input1);
    auto strides = conv->get_strides();
    auto paddings_begin = conv->get_pads_begin();
    auto paddings_end = conv->get_pads_end();
    auto dilations = conv->get_dilations();

    std::map<size_t, std::string> val_names{};

    auto element2val_name = [&val_names, &node_name](size_t element) {
        switch(element) {
            case 0: val_names[0] = "%const_zero"; break;
            case 1: val_names[1] = "%const_one"; break;
            default: val_names[element] = node_name + "_int_" + std::to_string(element); break;
        }
    };

    for(auto element : strides) {
        element2val_name(element);
    }
    for(auto element : paddings_begin) {
        element2val_name(element);
    }
    for(auto element : paddings_end) {
        element2val_name(element);
    }
    for(auto element : dilations) {
        element2val_name(element);
    }
    for(auto& val_name : val_names) {
        if(val_name.first == 0 || val_name.first == 1)
            continue;
        ss << "    " << val_name.second << " = torch.constant.int " << std::to_string(val_name.first) << "\n";
    }
    create_int_list(node_name, "strides", strides, ss);
    create_int_list(node_name, "paddings", std::vector<size_t>(paddings_begin.begin(), paddings_begin.end()), ss);
    create_int_list(node_name, "dilations", dilations, ss);

    ss << "    " << node_name;
    ss << " = \"torch.aten.conv1d\" (%" << input0_name << ", %" << input1_name << ", %none, "
       << node_name << "_strides, " << node_name << "_paddings, " << node_name << "_dilations, %const_one) : ("
       << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
       << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
       << "!torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int)";
    genOutputTypes(ss, node);
    ss << "\n";
}


static void translator_convolution_2d(const std::shared_ptr<const ov::op::v1::Convolution>& conv, std::ostream& ss) {
    const auto node = std::dynamic_pointer_cast<const ov::Node>(conv);
    auto& input0 = conv->input_value(0);
    auto& input1 = conv->input_value(1);
    auto node_name = std::string("%") + getMLIRName(node);
    auto input0_name = getMLIRName(input0);
    auto input1_name = getMLIRName(input1);
    auto strides = conv->get_strides();
    auto paddings_begin = conv->get_pads_begin();
    auto paddings_end = conv->get_pads_end();
    auto dilations = conv->get_dilations();
    std::map<size_t, std::string> val_names{};
    for(auto element : strides) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto element : paddings_begin) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto element : paddings_end) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto element : dilations) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto& val_name : val_names) {
        ss << "    " << val_name.second << " = torch.constant.int " << std::to_string(val_name.first) << "\n";
    }
    create_int_list(node_name, "strides", strides, ss);
    create_int_list(node_name, "paddings", std::vector<size_t>(paddings_begin.begin(), paddings_begin.end()), ss);
    create_int_list(node_name, "dilations", dilations, ss);

    ss << "    " << node_name;
    ss << " = \"torch.aten.conv2d\" (%" << input0_name << ", %" << input1_name << ", %none, "
       << node_name << "_strides, " << node_name << "_paddings, " << node_name << "_dilations, %const_one) : ("
       << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
       << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
       << "!torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int)";
    genOutputTypes(ss, node);
    ss << "\n";
}

static void translator_convolution_3d(const std::shared_ptr<const ov::op::v1::Convolution>& conv, std::ostream& ss) {
    const auto node = std::dynamic_pointer_cast<const ov::Node>(conv);
    auto& input0 = conv->input_value(0);
    auto& input1 = conv->input_value(1);
    auto node_name = std::string("%") + getMLIRName(node);
    auto input0_name = getMLIRName(input0);
    auto input1_name = getMLIRName(input1);
    auto strides = conv->get_strides();
    auto paddings_begin = conv->get_pads_begin();
    auto paddings_end = conv->get_pads_end();
    auto dilations = conv->get_dilations();
    std::map<size_t, std::string> val_names{};
    for(auto element : strides) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto element : paddings_begin) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto element : paddings_end) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto element : dilations) {
        val_names[element] = node_name + "_int_" + std::to_string(element);
    }
    for(auto& val_name : val_names) {
        ss << "    " << val_name.second << " = torch.constant.int " << std::to_string(val_name.first) << "\n";
    }
    create_int_list(node_name, "strides", strides, ss);
    create_int_list(node_name, "paddings", std::vector<size_t>(paddings_begin.begin(), paddings_begin.end()), ss);
    create_int_list(node_name, "dilations", dilations, ss);

    ss << "    " << node_name;
    ss << " = \"torch.aten.conv3d\" (%" << input0_name << ", %" << input1_name << ", %none, "
       << node_name << "_strides, " << node_name << "_paddings, " << node_name << "_dilations, %const_one) : ("
       << "!torch.vtensor<" << input0.get_partial_shape() << "," << ov_to_mlir_type(input0.get_element_type()) << ">, "
       << "!torch.vtensor<" << input1.get_partial_shape() << "," << ov_to_mlir_type(input1.get_element_type()) << ">, "
       << "!torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int)";
    genOutputTypes(ss, node);
    ss << "\n";
}

static void translator_convolution(const std::shared_ptr<const ov::Node>& node, std::ostream& ss) {
     auto conv = std::dynamic_pointer_cast<const ov::op::v1::Convolution>(node);
     if (!conv) return;
     auto& input0 = conv->input_value(0);
     switch(input0.get_partial_shape().size()) {
        case 3:
            translator_convolution_1d(conv, ss);
            break;
        case 4:
            translator_convolution_2d(conv, ss);
            break;
        case 5:
            translator_convolution_3d(conv, ss);
            break;
        default:
            std::string msg = std::string("Unsupported convolution for shape: ") + input0.get_partial_shape().to_string();
            OPENVINO_THROW(msg);
            break;
     }
}

EMIT_REG("Convolution", translator_convolution);
