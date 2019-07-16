#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <cstring>
#include "cpu_methods.h"
#include <stdio.h>

namespace py = pybind11;

int py2c(py::array py, struct c_array *c) {
    py::buffer_info info = py.request();
    char format[8];
    strcpy(format, info.format.c_str());
    c->ptr = info.ptr;
    c->itemsize = info.itemsize;
    c->size = info.size;
    c->format = format;
    c->ndim = info.ndim;
    c->shape = &info.shape[0];
    c->strides = &info.strides[0];
    return 1;
}

int makeIntNative_CPU(py::array input) {
    struct c_array array_struct;
    py2c(input, &array_struct);
    if (!checkInt_CPU(&array_struct)) {
        throw std::invalid_argument("Argument must be an int array");
    }
    if (!makeNative_CPU(&array_struct)) {
        throw std::invalid_argument("Error in cpu_methods.h::makeNative_CPU");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int8_t* max) {
    struct c_array array_struct;
    py2c(input, &array_struct);
    if (!getMax_8bit(&array_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_8bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int16_t* max) {
    struct c_array array_struct;
    py2c(input, &array_struct);
    if (!getMax_16bit(&array_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_16bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int32_t* max) {
    struct c_array array_struct;
    py2c(input, &array_struct);
    if (!getMax_32bit(&array_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_32bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int64_t* max) {
    struct c_array array_struct;
    py2c(input, &array_struct);
    if (!getMax_64bit(&array_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_64bit");
    }
    return 1;
}
