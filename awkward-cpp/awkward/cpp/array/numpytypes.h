#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cinttypes>
#include <stdexcept>
#include <complex>
#include "util.h"
#include "any.h"
#include "cpu_pybind11.h"
#include "cpu_methods.h"

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

    py::object getitem_tuple(py::tuple input, ssize_t index = 0, ssize_t select_index = -1) {
        if (index >= (ssize_t)input.size()) {
            return this->unwrap();
        }
        throw std::domain_error("getitem is not allowed on a NumpyScalar");
    }

    py::object unwrap() { return py::cast(thisScalar); }

    py::object tolist() { return unwrap(); }

    std::string str() {
        return py::str(unwrap());
    }

    std::string repr() {
        std::stringstream stream;
        stream << std::hex << (long)this;
        return "<NumpyScalar " + str() + " at 0x" + stream.str() + ">";
    }
};

class NumpyArray : public AnyArray {
public:
    virtual py::buffer_info request()                = 0;
};

NumpyArray* getNumpyArray_t(py::array input);

template <typename T>
class NumpyArray_t : public NumpyArray {
private:
    py::array_t<T> thisArray;

public:
    py::object unwrap() { return thisArray; }

    std::string str() {
        return py::str(thisArray);
    }

    std::string repr() {
        std::stringstream stream;
        stream << std::hex << (long)this;
        return "<NumpyArray " + str() + " at 0x" + stream.str() + ">";
    }

    AnyArray* deepcopy() {
        return new NumpyArray_t<T>(pyarray_deepcopy(thisArray));
    }

    ssize_t len() {
        return thisArray.request().size;
    }

    AnyArray* getitem(ssize_t start, ssize_t length, ssize_t step = 1) {
        if (length == 0) {
            auto newPyarray = py::array_t<T>(0);
            return new NumpyArray_t<T>(newPyarray);
        }
        return new NumpyArray_t<T>(slice_numpy(thisArray, start, length, step));
    }

    AnyOutput* getitem(ssize_t i) {
        int N = thisArray.request().strides[0] / thisArray.request().itemsize;
        return new NumpyScalar_t<T>(((T*)thisArray.request().ptr)[i * N]);
    }

    AnyArray* boolarray_getitem(py::array input) {
        ssize_t length = input.request().size;
        if (length != len()) {
            throw std::invalid_argument("bool array length must be equal to array length");
        }
        py::list temp;
        auto array_ptr = (bool*)input.request().ptr;
        for (ssize_t i = 0; i < length; i++) {
            if (array_ptr[i]) {
                temp.append(getitem(i)->unwrap());
            }
        }
        py::array_t<T> out = temp.cast<py::array_t<T>>();
        return getNumpyArray_t(out);
    }

    AnyArray* intarray_getitem(py::array input) {
        makeIntNative_CPU(input);
        input = input.cast<py::array_t<ssize_t>>();
        py::buffer_info array_info = input.request();
        auto array_ptr = (ssize_t*)array_info.ptr;

        auto out = py::array_t<T>(array_info.size);
        auto out_ptr = (T*)out.request().ptr;

        int N = thisArray.request().strides[0] / thisArray.request().itemsize;

        for (ssize_t i = 0; i < array_info.size; i++) {
            ssize_t here = array_ptr[i];
            if (here < 0 || here >= len()) {
                throw std::invalid_argument("int array indices must be within the bounds of the array");
            }
            out_ptr[i] = ((T*)thisArray.request().ptr)[here * N];
        }
        return getNumpyArray_t(out);
    }

    AnyArray* getitem(py::array input) {
        if (input.request().format.find("?") != std::string::npos) {
            return boolarray_getitem(input);
        }
        return intarray_getitem(input);
    }

    py::object getitem_tuple(py::tuple input, ssize_t index = 0, ssize_t select_index = -1) {
        if (index >= (ssize_t)input.size()) {
            return unwrap();
        }
        try {
            ssize_t here = input[index].cast<ssize_t>();
            return getitem(here)->getitem_tuple(input, index + 1, select_index);
        }
        catch(py::cast_error e) { }
        try {
            py::tuple check = input[index].cast<py::tuple>();
            check[0];
        }
        catch (std::exception e) {
            try {
                py::slice here = input[index].cast<py::slice>();
                size_t start, stop, step, slicelength;
                if (!here.compute(len(), &start, &stop, &step, &slicelength)) {
                    throw py::error_already_set();
                }
                if (index != input.size() - 1) {
                    throw std::invalid_argument("NumpyArray does not support object types");
                }
                return getitem((ssize_t)start, (ssize_t)slicelength, (ssize_t)step)->unwrap();
            }
            catch (py::cast_error e) {
                throw std::invalid_argument("argument index for __getitem__(tuple) not recognized");
            }
        }
        py::array here = input[index].cast<py::array>();
        if (select_index < 0) {
            if (index != input.size() - 1) {
                throw std::invalid_argument("NumpyArray does not support object types");
            }
            return getitem(here)->unwrap();
        }

        py::array_t<ssize_t> indices;
        if (here.request().format.find("?") != std::string::npos) {
            if (here.request().size != len()) {
                throw std::domain_error("Error: boolean array length is "
                + std::to_string(here.request().size) + ", but dimension length is "
                + std::to_string(len()) + ".");
            }
            py::list trues;
            for (ssize_t i = 0; i < here.request().size; i++) {
                if (((bool*)here.request().ptr)[i]) {
                    trues.append(i);
                }
            }
            indices = trues.cast<py::array_t<ssize_t>>();
        }
        else {
            try {
                indices = here.cast<py::array_t<ssize_t>>();
            }
            catch (py::cast_error e) {
                throw std::invalid_argument("array must be of bool or int type");
            }
        }
        if (indices.request().size == 1) {
            return getitem(((ssize_t*)indices.request().ptr)[0])->getitem_tuple(input, index + 1, select_index);
        }
        if (select_index > indices.request().size) {
            throw std::domain_error("Error: selection index exceeded selection length");
        }
        return getitem(((ssize_t*)indices.request().ptr)[select_index])->getitem_tuple(input, index + 1, select_index);
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
