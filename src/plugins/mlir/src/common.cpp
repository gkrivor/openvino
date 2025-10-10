#include "common.hpp"

namespace ov {
namespace mlir {

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

void genInputNames(std::ostream& ss, const std::shared_ptr<ov::Node>& node) {
    ss << "(";
    std::string sep;
    for (const auto& io : node->inputs()) {
        ss << std::exchange(sep, ", ")
           << "%" << io.get_friendly_name();
    }
    ss << ")";
}

void genInputTypes(std::ostream& ss, const std::shared_ptr<ov::Node>& node) {
    ss << "(";
    getVtensorTypeList(ss, node->inputs());
    ss << ")";
}

void genOutputTypes(std::ostream& ss, const std::shared_ptr<ov::Node>& node) {
    // Insert " -> " only if inputs size > 0
    if (!node->inputs().empty()) {
        ss << " -> ";
    }
    genVtensorTypeList(ss, node->outputs());
}

}
}