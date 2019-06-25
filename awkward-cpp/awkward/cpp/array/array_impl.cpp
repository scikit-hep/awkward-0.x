#include "_jagged.h"

PYBIND11_MODULE(array_impl, m) {
    py::class_<JaggedArray>(m, "JaggedArray")
        .def(py::init<py::array, py::array, py::object>())
        .def_property("starts", &JaggedArray::get_starts, &JaggedArray::set_starts)
        .def_property("stops", &JaggedArray::get_stops, &JaggedArray::set_stops)
        .def_property("content", &JaggedArray::get_content, &JaggedArray::set_content)
        .def_static("offsets2parents", &JaggedArray::offsets2parents)
        .def_static("counts2offsets", &JaggedArray::counts2offsets)
        .def_static("startsstops2parents", &JaggedArray::startsstops2parents)
        .def_static("parents2startsstops", &JaggedArray::parents2startsstops)
        .def_static("uniques2offsetsparents", &JaggedArray::uniques2offsetsparents)
        .def("__getitem__", (AnyArray* (JaggedArray::*)(ssize_t)) &JaggedArray::getitem)
        .def("__getitem__", (AnyArray* (JaggedArray::*)(ssize_t, ssize_t)) &JaggedArray::getitem)
        .def("__str__", &JaggedArray::str)
        .def("__len__", &JaggedArray::len);
}
