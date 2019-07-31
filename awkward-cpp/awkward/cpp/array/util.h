#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <stdexcept>
#include "cpu_methods.h"
#include "cpu_pybind11.h"

namespace py = pybind11;

template <typename T>
py::array_t<T> slice_numpy(py::array_t<T> input, ssize_t start, ssize_t length, ssize_t step = 1) {
    ssize_t arrayLen = input.request().size;
    if (step == 0) {
        throw std::invalid_argument("slice step cannot be 0");
    }
    if (length < 0) {
        throw std::invalid_argument("slice length cannot be less than 0");
    }
    if (start < 0 || start > arrayLen || start + (length * step) > arrayLen || start + (length * step) < -1) {
        throw std::out_of_range("slice must be in the bounds of the array");
    }
    py::buffer_info temp_info = py::buffer_info(input.request());
    temp_info.ptr = (void*)((T*)(input.request().ptr) + start);
    temp_info.size = length;
    temp_info.strides[0] = temp_info.strides[0] * step;
    temp_info.shape[0] = temp_info.size;
    return py::array_t<T>(temp_info);
}

template <typename T>
py::array_t<T> pyarray_deepcopy(py::array_t<T> input) {
    py::buffer_info input_info = input.request();
    struct c_array input_struct = py2c(&input_info);
    auto newArray = py::array_t<T>(input_info.size);
    py::buffer_info newArray_info = newArray.request();
    struct c_array newArray_struct = py2c(&newArray_info);

    newArray.resize(input_info.shape);
    if (!deepcopy_CPU(&newArray_struct, &input_struct)) {
        throw std::invalid_argument("Error in cpu_methods.h::deepcopy_CPU");
    }
    return newArray;
}
