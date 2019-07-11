#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cinttypes>
#include <stdexcept>
#include <complex>
#include "CPU_methods.h"
#include "numpytypes.h"

C_array_64 numpy2c(NumpyArray_t<std::int64_t> *input) {
    struct C_array_64 temp = {
        (std::int64_t*)input->request().ptr,
        8,
        input->request().size,
        input->request().format.c_str(),
        input->request().strides[0]
    };
    return temp;
}

C_array_32 numpy2c(NumpyArray_t<std::int32_t> *input) {
    struct C_array_32 temp = {
        (std::int32_t*)input->request().ptr,
        4,
        input->request().size,
        input->request().format.c_str(),
        input->request().strides[0]
    };
    return temp;
}

C_array_16 numpy2c(NumpyArray_t<std::int16_t> *input) {
    struct C_array_16 temp = {
        (std::int16_t*)input->request().ptr,
        2,
        input->request().size,
        input->request().format.c_str(),
        input->request().strides[0]
    };
    return temp;
}

C_array_8 numpy2c(NumpyArray_t<std::int8_t> *input) {
    struct C_array_8 temp = {
        (std::int8_t*)input->request().ptr,
        1,
        input->request().size,
        input->request().format.c_str(),
        input->request().strides[0]
    };
    return temp;
}

int makeNative_CPU(struct C_array_8 *input) {
    return 1;
}

int makeNative_CPU(struct C_array_16 *input) {
    return makeNative_16bit(input);
}

int makeNative_CPU(struct C_array_32 *input) {
    return makeNative_32bit(input);
}

int makeNative_CPU(struct C_array_64 *input) {
    return makeNative_64bit(input);
}

int makeIntNative_CPU(NumpyArray* input) {
    if (!isInt(input->request().format.c_str()))
        throw std::invalid_argument("argument must be of type int");
    return makeNative_CPU(numpy2c(getNumpyArray_t(input->unwrap())));
}

int checkunsigned2signed_CPU(struct C_array_8 *input) {
    return checkunsigned2signed_8bit(input);
}

int checkunsigned2signed_CPU(struct C_array_16 *input) {
    return checkunsigned2signed_16bit(input);
}

int checkunsigned2signed_CPU(struct C_array_32 *input) {
    return checkunsigned2signed_32bit(input);
}

int checkunsigned2signed_CPU(struct C_array_64 *input) {
    return checkunsigned2signed_64bit(input);
}

int checkPos_CPU(struct C_array_8 *input) {
    return checkPos_8bit(input);
}

int checkPos_CPU(struct C_array_16 *input) {
    return checkPos_16bit(input);
}

int checkPos_CPU(struct C_array_32 *input) {
    return checkPos_32bit(input);
}

int checkPos_CPU(struct C_array_64 *input) {
    return checkPos_64bit(input);
}

int checkPos_CPU(NumpyArray *input) {
    return checkPos_CPU(numpy2c(getNumpyArray_t(input->unwrap())));
}

int offsets2parents_CPU(C_array_64 *offsets, C_array_64 *parents) {
    return offsets2parents_int64(offsets, parents);
}

int offsets2parents_CPU(C_array_32 *offsets, C_array_32 *parents) {
    return offsets2parents_int32(offsets, parents);
}
