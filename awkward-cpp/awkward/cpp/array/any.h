#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>

namespace py = pybind11;

class AnyOutput {
public:
    virtual AnyOutput*  getitem(ssize_t)                             = 0;
    virtual py::object  unwrap()                                     = 0;
    virtual std::string str()                                        = 0;
    virtual py::object  tolist()                                     = 0;
};

class AnyArray : public AnyOutput {
public:
    virtual ssize_t     len()                                        = 0;
    virtual AnyArray*   getitem(ssize_t a, ssize_t b, ssize_t c = 1) = 0;
    virtual AnyArray*   deepcopy()                                   = 0;
};

class AwkwardArray : public AnyArray {
public:

};
