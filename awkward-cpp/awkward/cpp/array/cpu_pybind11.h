#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <cstring>
#include "cpu_methods.h"

namespace py = pybind11;

struct c_array py2c(py::buffer_info *info) {
    struct c_array out = {
        info->ptr,
        info->itemsize,
        info->size,
        info->format.c_str(),
        info->ndim,
        &info->shape[0],
        &info->strides[0],
    };
    return out;
}

int makeIntNative_CPU(py::array input) {
    py::buffer_info input_info = input.request();
    struct c_array input_struct = py2c(&input_info);
    /*if (!checkInt_CPU(&input_struct)) {
        throw std::invalid_argument("Argument must be an int array");
    }*/
    if (!makeNative_CPU(&input_struct)) {
        throw std::invalid_argument("Error in cpu_methods.h::makeNative_CPU");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int8_t* max) {
    py::buffer_info input_info = input.request();
    struct c_array input_struct = py2c(&input_info);
    if (!getMax_8bit(&input_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_8bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int16_t* max) {
    py::buffer_info input_info = input.request();
    struct c_array input_struct = py2c(&input_info);
    if (!getMax_16bit(&input_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_16bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int32_t* max) {
    py::buffer_info input_info = input.request();
    struct c_array input_struct = py2c(&input_info);
    if (!getMax_32bit(&input_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_32bit");
    }
    return 1;
}

int getMax_CPU(py::array input, std::int64_t* max) {
    py::buffer_info input_info = input.request();
    struct c_array input_struct = py2c(&input_info);
    if (!getMax_64bit(&input_struct, 0, 0, max)) {
        throw std::invalid_argument("Error in cpu_methods.h::getMax_64bit");
    }
    return 1;
}
