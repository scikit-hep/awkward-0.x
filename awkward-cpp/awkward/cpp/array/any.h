#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>

namespace py = pybind11;

class AnyOutput {
public:

};

class AnyArray : public AnyOutput {
public:
    virtual std::string str()                     = 0;
    virtual ssize_t     len()                     = 0;
    virtual AnyArray*   getitem(ssize_t)          = 0;
    virtual AnyArray*   getitem(ssize_t, ssize_t) = 0;
};

class AwkwardArray : public AnyArray {
public:

};
