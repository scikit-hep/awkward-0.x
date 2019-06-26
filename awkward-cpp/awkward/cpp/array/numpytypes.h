#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cinttypes>
#include <stdexcept>
#include <complex>
#include "util.h"
#include "any.h"

namespace py = pybind11;

class NumpyScalar : public AnyOutput {
public:

};

class NumpyArray : public AnyArray {
private:
    py::array thisArray;

    template <typename T>
    std::string toString(bool comma = false) {
        py::buffer_info array_info = thisArray.request();
        auto array_ptr = (T*)array_info.ptr;
        int N = array_info.strides[0] / array_info.itemsize;

        std::string out;
        out.reserve(array_info.size * 10);
        ssize_t len = array_info.size;
        out.append("[");
        for (ssize_t i = 0; i < len; i++) {
            if (i != 0) {
                if (comma) {
                    out.append(",");
                }
                out.append(" ");
            }
            out.append(convert_to_string(array_ptr[i * N]));
        }
        out.append("]");
        out.shrink_to_fit();
        return out;
    }

public:
    py::object unwrap() { return thisArray; }

    std::string str() {
        std::string format = thisArray.request().format;
        if (format.find("q") != std::string::npos)
            return toString<std::int64_t>();
        if (format.find("Q") != std::string::npos)
            return toString<std::uint64_t>();
        if (format.find("l") != std::string::npos)
            return toString<std::int32_t>();
        if (format.find("L") != std::string::npos)
            return toString<std::uint32_t>();
        if (format.find("h") != std::string::npos)
            return toString<std::int16_t>();
        if (format.find("H") != std::string::npos)
            return toString<std::uint16_t>();
        if (format.find("b") != std::string::npos)
            return toString<std::int8_t>();
        if (format.find("B") != std::string::npos)
            return toString<std::uint8_t>();
        /*if (format.find("w") != std::string::npos)
            return toString<std::string>();
        if (format.find("?") != std::string::npos)
            return toString<bool>();*/
        if (format.find("Zf") != std::string::npos)
            return toString<std::complex<float>>();
        if (format.find("Zd") != std::string::npos)
            return toString<std::complex<double>>();
        if (format.find("f") != std::string::npos)
            return toString<float>();
        if (format.find("d") != std::string::npos)
            return toString<double>();

        throw std::invalid_argument("[__str__ is not supported for this type]");
    }

    ssize_t len() {
        return thisArray.request().size;
    }

    AnyArray* getitem(ssize_t start, ssize_t end) {
        if (start < 0) {
            start += thisArray.request().size;
        }
        if (end < 0) {
            end += thisArray.request().size;
        }
        if (start < 0 || start > end || end > thisArray.request().size) {
            throw std::out_of_range("getitem must be in the bounds of the array");
        }
        py::buffer_info temp_info = py::buffer_info();
        std::string format = thisArray.request().format;
        if (format.find("q") != std::string::npos)
            temp_info.ptr = (void*)((std::int64_t*)(thisArray.request().ptr) + start);
        else if (format.find("Q") != std::string::npos)
            temp_info.ptr = (void*)((std::uint64_t*)(thisArray.request().ptr) + start);
        else if (format.find("L") != std::string::npos)
            temp_info.ptr = (void*)((std::uint32_t*)(thisArray.request().ptr) + start);
        else if (format.find("h") != std::string::npos)
            temp_info.ptr = (void*)((std::int16_t*)(thisArray.request().ptr) + start);
        else if (format.find("H") != std::string::npos)
            temp_info.ptr = (void*)((std::uint16_t*)(thisArray.request().ptr) + start);
        else if (format.find("b") != std::string::npos)
            temp_info.ptr = (void*)((std::int8_t*)(thisArray.request().ptr) + start);
        else if (format.find("B") != std::string::npos)
            temp_info.ptr = (void*)((std::uint8_t*)(thisArray.request().ptr) + start);
        /*if (format.find("w") != std::string::npos)
            temp_info.ptr = (void*)((std::string*)(thisArray.request().ptr) + start);*/
        else if (format.find("?") != std::string::npos)
            temp_info.ptr = (void*)((bool*)(thisArray.request().ptr) + start);
        else if (format.find("Zf") != std::string::npos)
            temp_info.ptr = (void*)((std::complex<float>*)(thisArray.request().ptr) + start);
        else if (format.find("Zd") != std::string::npos)
            temp_info.ptr = (void*)((std::complex<double>*)(thisArray.request().ptr) + start);
        else if (format.find("f") != std::string::npos)
            temp_info.ptr = (void*)((float*)(thisArray.request().ptr) + start);
        else if (format.find("d") != std::string::npos)
            temp_info.ptr = (void*)((double*)(thisArray.request().ptr) + start);
        else
            temp_info.ptr = (void*)((std::int32_t*)(thisArray.request().ptr) + start);
        temp_info.itemsize = thisArray.request().itemsize;
        temp_info.size = end - start;
        temp_info.format = thisArray.request().format;
        temp_info.ndim = thisArray.request().ndim;
        temp_info.strides = thisArray.request().strides;
        temp_info.shape = thisArray.request().shape;
        temp_info.shape[0] = temp_info.size;
        return new NumpyArray(py::array(temp_info));
    }

    AnyArray* getitem(ssize_t e) {
        throw std::invalid_argument("getitem(ssize_t) is not yet implemented in NumpyArray");
        return this;
    }

    NumpyArray(py::array input) { thisArray = input; }

    py::buffer_info request() { return thisArray.request(); }
};
