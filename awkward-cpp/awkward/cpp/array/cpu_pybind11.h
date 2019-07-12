#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <cstring>
#include "cpu_methods.h"
#include <stdio.h>

namespace py = pybind11;

struct c_array py2c(py::array input) {
    char format[8];
    strcpy(format, input.request().format.c_str());
    struct c_array out = {
        input.request().ptr,
        input.request().itemsize,
        input.request().size,
        format,
        input.request().ndim,
        &input.request().shape[0],
        &input.request().strides[0]
    };
    return out;
}

int makeIntNative_CPU(py::array input) {
    if (!checkInt_CPU(&py2c(input))) {
        throw std::invalid_argument("Argument must be an int array");
    }
    if (!makeNative_CPU(&py2c(input))) {
        throw std::invalid_argument("Error in cpu_methods.h::makeNative_CPU");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int8_t* max) {
    if (!getMax_8bit(&py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_8bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int16_t* max) {
    if (!getMax_16bit(&py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_16bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int32_t* max) {
    if (!getMax_32bit(&py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_32bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int64_t* max) {
    if (!getMax_64bit(&py2c(input), 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_64bit");
    }
    return 1;
}
