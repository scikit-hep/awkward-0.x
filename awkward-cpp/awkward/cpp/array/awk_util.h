#ifndef AWK_UTIL_H
#define AWK_UTIL_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cinttypes>
#include <stdexcept>
#include <complex>

namespace py = pybind11;

std::uint16_t swap_uint16(std::uint16_t val) {
    return (val << 8) | (val >> 8);
}

std::int16_t swap_int16(std::int16_t val) {
    return (val << 8) | ((val >> 8) & 0xFF);
}

std::uint32_t swap_uint32(std::uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

std::int32_t swap_int32(std::int32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | ((val >> 16) & 0xFFFF);
}

std::uint64_t swap_uint64(std::uint64_t val) {
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL) | ((val >> 8) & 0x00FF00FF00FF00FFULL);
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL) | ((val >> 16) & 0x0000FFFF0000FFFFULL);
    return (val << 32) | (val >> 32);
}

std::int64_t swap_int64(std::int64_t val) {
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL) | ((val >> 8) & 0x00FF00FF00FF00FFULL);
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL) | ((val >> 16) & 0x0000FFFF0000FFFFULL);
    return (val << 32) | ((val >> 32) & 0xFFFFFFFFULL);
}

bool isNative(py::array input) {
    char ch = input.request().format.at(0);
    union {
        uint32_t i;
        char c[4];
    } bint = { 0x01020304 };
    return ((bint.c[0] == 1 && ch != '<') || (bint.c[0] != 1 && ch != '>'));
}

bool isNativeInt(py::array input) {
    std::string intList = "qQlLhHbB";
    if (intList.find(input.request().format.at(0)) == std::string::npos) {
        throw std::invalid_argument("argument must be of type int");
    }
    return isNative(input);
}

void makeNative(py::array_t<std::uint8_t> input) { return; }


void makeNative(py::array_t<std::int8_t> input) { return; }


void makeNative(py::array_t<std::uint16_t> input) {

    if (isNative(input)) {
        return;
    }
    py::buffer_info array_info = input.request();
    auto array_ptr = (std::uint16_t*)array_info.ptr;
    int N = array_info.shape[0] / array_info.itemsize;

    for (ssize_t i = 0; i < array_info.size; i++) {
        array_ptr[i * N] = swap_uint16(array_ptr[i * N]);
    }
    return;
}


void makeNative(py::array_t<std::int16_t> input) {

    if (isNative(input)) {
        return;
    }
    py::buffer_info array_info = input.request();
    auto array_ptr = (std::int16_t*)array_info.ptr;
    int N = array_info.shape[0] / array_info.itemsize;

    for (ssize_t i = 0; i < array_info.size; i++) {
        array_ptr[i * N] = swap_int16(array_ptr[i * N]);
    }
    return;
}


void makeNative(py::array_t<std::uint32_t> input) {

    if (isNative(input)) {
        return;
    }
    py::buffer_info array_info = input.request();
    auto array_ptr = (std::uint32_t*)array_info.ptr;
    int N = array_info.shape[0] / array_info.itemsize;

    for (ssize_t i = 0; i < array_info.size; i++) {
        array_ptr[i * N] = swap_uint32(array_ptr[i * N]);
    }
    return;
}


void makeNative(py::array_t<std::int32_t> input) {

    if (isNative(input)) {
        return;
    }
    py::buffer_info array_info = input.request();
    auto array_ptr = (std::int32_t*)array_info.ptr;
    int N = array_info.shape[0] / array_info.itemsize;

    for (ssize_t i = 0; i < array_info.size; i++) {
        array_ptr[i * N] = swap_int32(array_ptr[i * N]);
    }
    return;
}


void makeNative(py::array_t<std::uint64_t> input) {

    if (isNative(input)) {
        return;
    }
    py::buffer_info array_info = input.request();
    auto array_ptr = (std::uint64_t*)array_info.ptr;
    int N = array_info.shape[0] / array_info.itemsize;

    for (ssize_t i = 0; i < array_info.size; i++) {
        array_ptr[i * N] = swap_uint64(array_ptr[i * N]);
    }
    return;
}

void makeNative(py::array_t<std::int64_t> input) {

    if (isNative(input)) {
        return;
    }
    py::buffer_info array_info = input.request();
    auto array_ptr = (std::int64_t*)array_info.ptr;
    int N = array_info.shape[0] / array_info.itemsize;

    for (ssize_t i = 0; i < array_info.size; i++) {
        array_ptr[i * N] = swap_int64(array_ptr[i * N]);
    }
    return;
}

void makeIntNative(py::array input) {
    if (isNativeInt(input)) {
        return;
    }
    char code = input.request().format.at(1);
    if (code == 'q') {
        makeNative(input.cast<py::array_t<std::int64_t>>());
        return;
    }
    if (code == 'Q') {
        makeNative(input.cast<py::array_t<std::uint64_t>>());
        return;
    }
    if (code == 'l') {
        makeNative(input.cast<py::array_t<std::int32_t>>());
        return;
    }
    if (code == 'L') {
        makeNative(input.cast<py::array_t<std::uint32_t>>());
        return;
    }
    if (code == 'h') {
        makeNative(input.cast<py::array_t<std::int16_t>>());
        return;
    }
    if (code == 'H') {
        makeNative(input.cast<py::array_t<std::uint16_t>>());
        return;
    }
    throw std::invalid_argument("argument must be of type int");
}

std::string convert_to_string(std::int64_t val) {
    return std::to_string(val);
}

std::string convert_to_string(std::uint64_t val) {
    return std::to_string(val);
}

std::string convert_to_string(std::int32_t val) {
    return std::to_string(val);
}

std::string convert_to_string(std::uint32_t val) {
    return std::to_string(val);
}

std::string convert_to_string(std::int16_t val) {
    return std::to_string(val);
}

std::string convert_to_string(std::uint16_t val) {
    return std::to_string(val);
}

std::string convert_to_string(std::int8_t val) {
    return std::to_string((std::int16_t)val);
}

std::string convert_to_string(std::uint8_t val) {
    return std::to_string((std::uint16_t)val);
}

std::string convert_to_string(std::string val) {
    return val;
}


#endif