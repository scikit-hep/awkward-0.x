#include <iostream>

#include <pybind11/pybind11.h>

namespace py = pybind11;

// // Temporary test variables
// int offsets[5] = { 0, 2, 4, 4, 7 };
// int offsetLen = 5;
// int parents[7] = { 0, 0, 0, 0, 0, 0, 0 };
// int parentLen = 7;

void offsets2parents() {
    std::cout << "hello world" << std::endl;

    // int j = 0;
    // int k = -1;
    // for (int i = 0; i < offsetLen; i++) {
    //     while (j < offsets[i]) {
    //         parents[j] = k;
    //         j += 1;
    //     }
    //     k += 1;
    // }
}

std::string toString(int input[], int length) {
    std::string str = "{ ";
    for (int i = 0; i < length - 1; i++) {
        str += std::to_string(input[i]);
        str += ", ";
    }
    if (length > 0) {
        str += std::to_string(input[length - 1]);
        str += " ";
    }
    str += "}";
    return str;
}

// std::string showOffsets() {
//     return toString(offsets, offsetLen);
// }

// std::string showParents() {
//     return toString(parents, parentLen);
// }

PYBIND11_MODULE(_jagged, m) {
    m.doc() = "Pybind11 functions for jagged arrays";

    m.def("offsets2parents", &offsets2parents, "Populates a properly-sized parents array based on the values from an offsets array.");
}
