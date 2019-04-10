#include <pybind11/pybind11.h>

namespace py = pybind11;

// Temporary test variables
int offsets[5] = { 0, 2, 4, 4, 7 };
int offsetLen = 5;
int parents[7] = { 0, 0, 0, 0, 0, 0, 0 };
int parentLen = 7;

// Function Name: offsets2parents
// Description: Populates a properly-sized parents
//              array based on the values from an
//              offsets array.
// Inputs:      Nothing technically, until I find out
//              how to pass arrays by reference and
//              modify their contents in a function
// Outputs:     -- (array should be modified)
// Note:        I modeled this function after
//              offsets2parents_fill() in the jagged
//              file from awkward-numba

void offsets2parents() {
    int j = 0;
    int k = -1;
    for (int i = 0; i < offsetLen; i++) {
        while (j < offsets[i]) {
            parents[j] = k;
            j += 1;
        }
        k += 1;
    }
}


// Function Name: toString
// Description:   Converts an array into a string in the
//                format '{ 1, 2, 3 }'
// Inputs:        int input[]        The array to be converted
//                int length        The length of the array
// Outputs:       string str        Converted product
// Note:          This is another temporary piece of the
//                file just to be used for test purposes

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

// Function Name: showOffsets
// Description:   Calls toString() for the offsets array
//                so it can be displayed in the python
//                shell
// Inputs:        --
// Outputs:       string offsets
// Note:          Still another test function

std::string showOffsets() {
    return toString(offsets, offsetLen);
}

// Function Name: showParents
// Description:   Calls toString() for the parents array
//                so it can be displayed in the python
//                shell
// Inputs:           --
// Outputs:       string parents
// Note:          Still another test function

std::string showParents() {
    return toString(parents, parentLen);
}

// Note:          This section is what binds the C++
//                code to python. It will normally be in
//                a separate file, but for convenience
//                and simplicity I've left it here.

PYBIND11_MODULE(JaggedArray, m) {
    m.doc() = "Pybind11 functions for jagged arrays";

    m.def("offsets2parents", &offsets2parents, "Populates a properly-sized parents array based on the values from an offsets array.");
    m.def("toString", &toString, "Converts an array into a string in the format \'{ 1, 2, 3 }\'");
    m.def("showOffsets", &showOffsets, "Calls toString() for the offsets array so it can be displayed in the python shell");
    m.def("showParents", &showParents, "Calls toString() for the parents array so it can be displayed in the python shell");
}
