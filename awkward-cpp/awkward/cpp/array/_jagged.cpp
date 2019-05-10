#include <cinttypes>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// // Temporary test variables
// int offsets[5] = { 0, 2, 4, 4, 7 };
// int offsetLen = 5;
// int parents[7] = { 0, 0, 0, 0, 0, 0, 0 };
// int parentLen = 7;

py::array_t<std::int64_t> offsets2parents_int64(py::array_t<std::int64_t> offsets) {
    py::buffer_info offsets_info = offsets.request();
    auto offsets_ptr = (std::int64_t*)offsets_info.ptr;

    size_t parents_length = offsets_ptr[offsets_info.size - 1];
    auto parents = py::array_t<std::int64_t>(parents_length);
    py::buffer_info parents_info = parents.request();

    auto parents_ptr = (std::int64_t*)parents_info.ptr;

    size_t j = 0;
    size_t k = -1;
    for (size_t i = 0; i < (size_t)offsets_info.size; i++) {
        while (j < (size_t)offsets_ptr[i]) {
            parents_ptr[j] = k;
            j += 1;
        }
        k += 1;
    }

    return parents;
}

// std::string toString(int input[], int length) {
//     std::string str = "{ ";
//     for (int i = 0; i < length - 1; i++) {
//         str += std::to_string(input[i]);
//         str += ", ";
//     }
//     if (length > 0) {
//         str += std::to_string(input[length - 1]);
//         str += " ";
//     }
//     str += "}";
//     return str;
// }

// std::string showOffsets() {
//     return toString(offsets, offsetLen);
// }

// std::string showParents() {
//     return toString(parents, parentLen);
// }

PYBIND11_MODULE(_jagged, m) {
    m.def("offsets2parents_int64", &offsets2parents_int64, "");
}
