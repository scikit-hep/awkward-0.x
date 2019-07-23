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

template <typename T>
class NumpyScalar_t : public NumpyScalar {
private:
    T thisScalar;

public:
    NumpyScalar_t<T>(T scalar) { thisScalar = scalar; }

    AnyOutput* getitem(ssize_t) {
        throw std::domain_error("getitem is not allowed on a NumpyScalar");
    }

    py::object unwrap() { return py::cast(thisScalar); }

    py::object tolist() { return unwrap(); }

    std::string str() {
        return py::str(unwrap());
    }
};

class NumpyArray : public AnyArray {
public:
    virtual py::buffer_info request() = 0;
};

template <typename T>
class NumpyArray_t : public NumpyArray {
private:
    py::array_t<T> thisArray;

public:
    py::object unwrap() { return thisArray; }

    std::string str() {
        return py::str(thisArray);
    }

    AnyArray* deepcopy() {
        return new NumpyArray_t<T>(pyarray_deepcopy(thisArray));
    }

    ssize_t len() {
        return thisArray.request().size;
    }

    AnyArray* getitem(ssize_t start, ssize_t length, ssize_t step = 1) {
        return new NumpyArray_t<T>(slice_numpy(thisArray, start, length, step));
    }

    AnyOutput* getitem(ssize_t i) {
        return new NumpyScalar_t<T>(((T*)thisArray.request().ptr)[i]);
    }

    py::object tolist() {
        py::list out;
        for (ssize_t i = 0; i < len(); i++) {
            out.append(getitem(i)->tolist());
        }
        return out;
    }

    NumpyArray_t<T>(py::array_t<T> input) { thisArray = input; }

    py::buffer_info request() { return thisArray.request(); }
};

NumpyArray* getNumpyArray_t(py::array input) {
    std::string format = input.request().format;
    if (format.find("q") != std::string::npos)
        return new NumpyArray_t<std::int64_t>(input.cast<py::array_t<std::int64_t>>());
    else if (format.find("Q") != std::string::npos)
        return new NumpyArray_t<std::uint64_t>(input.cast<py::array_t<std::uint64_t>>());
    else if (format.find("l") != std::string::npos)
        return new NumpyArray_t<std::int32_t>(input.cast<py::array_t<std::int32_t>>());
    else if (format.find("L") != std::string::npos)
        return new NumpyArray_t<std::uint32_t>(input.cast<py::array_t<std::uint32_t>>());
    else if (format.find("h") != std::string::npos)
        return new NumpyArray_t<std::int16_t>(input.cast<py::array_t<std::int16_t>>());
    else if (format.find("H") != std::string::npos)
        return new NumpyArray_t<std::uint16_t>(input.cast<py::array_t<std::uint16_t>>());
    else if (format.find("b") != std::string::npos)
        return new NumpyArray_t<std::int8_t>(input.cast<py::array_t<std::int8_t>>());
    else if (format.find("B") != std::string::npos)
        return new NumpyArray_t<std::uint8_t>(input.cast<py::array_t<std::uint8_t>>());
    else if (format.find("?") != std::string::npos)
        return new NumpyArray_t<bool>(input.cast<py::array_t<bool>>());
    else if (format.find("Zf") != std::string::npos)
        return new NumpyArray_t<std::complex<float>>(input.cast<py::array_t<std::complex<float>>>());
    else if (format.find("Zd") != std::string::npos)
        return new NumpyArray_t<std::complex<double>>(input.cast<py::array_t<std::complex<double>>>());
    else if (format.find("f") != std::string::npos)
        return new NumpyArray_t<float>(input.cast<py::array_t<float>>());
    else if (format.find("d") != std::string::npos)
        return new NumpyArray_t<double>(input.cast<py::array_t<double>>());
    else
        throw std::invalid_argument("array type not supported");
}
