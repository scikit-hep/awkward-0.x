#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <cstring>
#include "cpu_methods.h"

namespace py = pybind11;

struct c_array py2c(py::buffer_info info) {
    char format[6];
    strcpy(format, info.format.c_str());
    const ssize_t *shape = &info.shape[0];
    const ssize_t *strides = &info.strides[0];

    struct c_array out = {
        info.ptr,
        info.itemsize,
        info.size,
        format,
        info.ndim,
        shape,
        strides,
    };
    return out;
}

int makeIntNative_CPU(py::array input) {
    if (!checkInt_CPU(py2c(input.request()))) {
        throw std::invalid_argument("Argument must be an int array");
    }
    if (!makeNative_CPU(py2c(input.request()))) {
        throw std::invalid_argument("Error in cpu_methods.h::makeNative_CPU");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int8_t* max) {
    if (!getMax_8bit(py2c(input.request()), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_8bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int16_t* max) {
    if (!getMax_16bit(py2c(input.request()), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_16bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int32_t* max) {
    if (!getMax_32bit(py2c(input.request()), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_32bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int64_t* max) {
    if (!getMax_64bit(py2c(input.request()), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_64bit");
    }
    return 1;
}
