#include "jagged.h"

PYBIND11_MODULE(array_impl, m) {
    py::class_<JaggedArray>(m, "JaggedArray")
        .def(py::init<py::object, py::object, py::object>())
        .def_property("starts", &JaggedArray::get_starts, &JaggedArray::python_set_starts)
        .def_property("stops", &JaggedArray::get_stops, &JaggedArray::python_set_stops)
        .def_property("content", &JaggedArray::python_get_content, &JaggedArray::python_set_content)
        .def_static("offsets2parents", &JaggedArray::python_offsets2parents)
        .def_static("counts2offsets", &JaggedArray::python_counts2offsets)
        .def_static("startsstops2parents", &JaggedArray::python_startsstops2parents)
        .def_static("parents2startsstops", &JaggedArray::python_parents2startsstops,
            py::arg("parents"), py::arg("length") = -1)
        .def_static("uniques2offsetsparents", &JaggedArray::python_uniques2offsetsparents)
        .def_static("fromiter", &JaggedArray::fromiter)
        .def_static("fromoffsets", &JaggedArray::python_fromoffsets)
        .def_static("fromcounts", &JaggedArray::python_fromcounts)
        .def_static("fromparents", &JaggedArray::python_fromparents)
        .def_static("fromuniques", &JaggedArray::python_fromuniques)
        .def_static("fromjagged", &JaggedArray::fromjagged)
        .def("copy", &JaggedArray::copy)
        .def("deepcopy", &JaggedArray::deepcopy)
        .def("tolist", &JaggedArray::tolist)
        .def("__getitem__", (py::object (JaggedArray::*)(ssize_t)) &JaggedArray::python_getitem)
        .def("__getitem__", (py::object (JaggedArray::*)(py::slice)) &JaggedArray::python_getitem)
        .def("__getitem__", (py::object (JaggedArray::*)(py::array)) &JaggedArray::python_getitem)
        .def("__getitem__", (py::object (JaggedArray::*)(py::tuple)) &JaggedArray::python_getitem)
        .def("__getitem__", (py::object (JaggedArray::*)(JaggedArray*)) &JaggedArray::python_getitem)
        .def("__str__", &JaggedArray::str)
        .def("__len__", &JaggedArray::len)
        .def("__iter__", &JaggedArray::iter)
        .def("__repr__", &JaggedArray::repr);
    py::class_<JaggedArray::JaggedArrayIterator>(m, "JaggedArrayIterator")
        .def(py::init<JaggedArray*>())
        .def("__iter__", &JaggedArray::JaggedArrayIterator::iter)
        .def("__next__", &JaggedArray::JaggedArrayIterator::next);
}
