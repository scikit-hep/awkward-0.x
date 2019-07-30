#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>

namespace py = pybind11;

class AnyOutput {
public:

    virtual py::object  getitem_tuple(py::tuple i,
        ssize_t j = 0, ssize_t select_index = -1)          = 0;
    virtual py::object  unwrap()                           = 0;
    virtual std::string str()                              = 0;
    virtual std::string repr()                             = 0;
    virtual py::object  tolist()                           = 0;
};

class AnyArray : public AnyOutput {
public:
    virtual AnyOutput*  getitem(ssize_t)                   = 0;
    virtual AnyArray*   getitem(ssize_t a, ssize_t b,
        ssize_t c = 1)                                     = 0;
    virtual AnyArray*   getitem(py::array)                 = 0;
    virtual AnyArray*   deepcopy()                         = 0;
    virtual ssize_t     len()                              = 0;
    virtual AnyArray*   boolarray_getitem(py::array input) = 0;
    virtual AnyArray*   intarray_getitem(py::array input)  = 0;
};

class AwkwardArray : public AnyArray {
public:

};
