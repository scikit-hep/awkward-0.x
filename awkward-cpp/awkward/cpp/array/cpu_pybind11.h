#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <cstring>
#include "cpu_methods.h"
#include <stdio.h>

namespace py = pybind11;

struct c_array py2c(py::array input) {
    py::buffer_info info = input.request();

    if (info.ndim > 15) {
        throw std::invalid_argument("Array cannot exceed 15 dimensions");
    }

    char format[15];
    strcpy(format, info.format.c_str());
    ssize_t shape[15];
    std::copy(info.shape.begin(), info.shape.end(), shape);
    ssize_t strides[15];
    std::copy(info.strides.begin(), info.strides.end(), strides);

    struct c_array out = {
        info.ptr,
        info.itemsize,
        info.size,
        format,
        info.ndim,
        &shape[0],
        &strides[0]
    };
    return out;
}

int makeIntNative_CPU(py::array input) {
    if (!checkInt_CPU(py2c(input))) {
        throw std::invalid_argument("Argument must be an int array");
    }
    if (!makeNative_CPU(py2c(input))) {
        throw std::invalid_argument("Error in cpu_methods.h::makeNative_CPU");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int8_t* max) {
    if (!getMax_8bit(py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_8bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int16_t* max) {
    if (!getMax_16bit(py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_16bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int32_t* max) {
    if (!getMax_32bit(py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_32bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int64_t* max) {
    if (!getMax_64bit(py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_64bit");
    }
    return 1;
}
