#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cinttypes>
#include <stdexcept>
#include <complex>
#include <sstream>
#include <iomanip>

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
    if (start < 0 || start >= arrayLen || start + (length * step) > arrayLen || start + (length * step) < -1) {
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
    auto newArray = py::array_t<T>(input.request().size);
    auto newArray_ptr = (T*)newArray.request().ptr;
    auto input_ptr = (T*)input.request().ptr;
    int N = input.request().strides[0] / input.request().itemsize;
    for (ssize_t i = 0; i < input.request().size; i++) {
        newArray_ptr[i] = input_ptr[i * N];
    }
    return newArray;
}
