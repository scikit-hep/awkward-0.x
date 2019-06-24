/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
TODO:
= Implement NumpyScalar Class
    - Handle typing and printing of every type
    - Would be really useful to have a system of determining array 
        type from a method
    - Once this is done, can do getitem + iter in NumpyArray
= Deal with more array characteristics
    - Multidimensional arrays
    - Offset arrays
= Handle endianness in converting scalar types to strings
= Figure out how to separate a pybind11 project into multiple
    *.cpp files
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cinttypes>
#include <stdexcept>
#include <complex>
#include "awk_util.h"

namespace py = pybind11;

class ContentType {
public:
    virtual std::string  __str__()                     = 0;
    virtual std::string  __repr__()                    = 0;
    virtual ssize_t      __len__()                     = 0;
    virtual ContentType* __getitem__(ssize_t)          = 0;
    virtual ContentType* __getitem__(ssize_t, ssize_t) = 0;
};

class NumpyArray : public ContentType {
private:
    py::array thisArray;
    // ssize_t iter_index;

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
    std::string __str__() {
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

    std::string __repr__() {
        std::string format = thisArray.request().format;
        if (format.find("q") != std::string::npos)
            return "array(" + toString<std::int64_t>(true) + ")";
        if (format.find("Q") != std::string::npos)
            return "array(" + toString<std::uint64_t>(true) + ")";
        if (format.find("l") != std::string::npos)
            return "array(" + toString<std::int32_t>(true) + ")";
        if (format.find("L") != std::string::npos)
            return "array(" + toString<std::uint32_t>(true) + ")";
        if (format.find("h") != std::string::npos)
            return "array(" + toString<std::int16_t>(true) + ")";
        if (format.find("H") != std::string::npos)
            return "array(" + toString<std::uint16_t>(true) + ")";
        if (format.find("b") != std::string::npos)
            return "array(" + toString<std::int8_t>(true) + ")";
        if (format.find("B") != std::string::npos)
            return "array(" + toString<std::uint8_t>(true) + ")";
        /*if (format.find("w") != std::string::npos)
            return "array(" + toString<std::string>(true) + ")";
        if (format.find("?") != std::string::npos)
            return "array(" + toString<bool>(true) + ")";*/
        if (format.find("Zf") != std::string::npos)
            return "array(" + toString<std::complex<float>>(true) + ")";
        if (format.find("Zd") != std::string::npos)
            return "array(" + toString<std::complex<double>>(true) + ")";
        if (format.find("f") != std::string::npos)
            return "array(" + toString<float>(true) + ")";
        if (format.find("d") != std::string::npos)
            return "array(" + toString<double>(true) + ")";
        return "<NumpyArray [of unsupported array type]>";
    }

    ssize_t __len__() {
        return thisArray.request().size;
    }

    ContentType* __getitem__(ssize_t start, ssize_t end) {
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
        temp_info.format = format;
        temp_info.ndim = thisArray.request().ndim;
        temp_info.strides = thisArray.request().strides;
        temp_info.shape = thisArray.request().shape;
        temp_info.shape[0] = temp_info.size;
        return new NumpyArray(py::array(temp_info));
    }

    ContentType* __getitem__(ssize_t e) {
        throw std::invalid_argument("__getitem__(ssize_t) is not yet implemented in NumpyArray");
        return this;
    }

    /*ContentType* __iter__() {
        iter_index = 0;
        return this;
    }

    std::int64_t __next__() {
        if (iter_index >= __len__()) {
            throw py::stop_iteration();
        }
        return __getitem__(iter_index++);
    }*/

    NumpyArray(py::array input) { thisArray = input; }
    py::buffer_info request() { return thisArray.request(); }
    py::array get_array() { return thisArray; }
};

class AwkwardArray : public ContentType {
public:

};

class JaggedArraySrc : public AwkwardArray {
public:
    py::array_t<std::int64_t> starts,
                              stops;
    ContentType*              content;
    ssize_t                   iter_index;

    ContentType* get_content() { return content; }

    void set_content(py::object content_) {
        try {
            content = content_.cast<JaggedArraySrc*>();
            return;
        }
        catch (py::cast_error e) { }
        try {
            content = new NumpyArray(content_.cast<py::array>());
            return;
        }
        catch (py::cast_error e) {
            throw std::invalid_argument("Invalid type for JaggedArray.content");
        }
    }

    py::array_t<std::int64_t> get_starts() { return starts; }

    void set_starts(py::array starts_) {
        makeIntNative(starts_);
        starts_ = starts_.cast<py::array_t<std::int64_t>>();
        py::buffer_info starts_info = starts_.request();
        if (starts_info.ndim < 1) {
            throw std::domain_error("starts must have at least 1 dimension");
        }
        int N = starts_info.strides[0] / starts_info.itemsize;
        auto starts_ptr = (std::int64_t*)starts_info.ptr;
        for (ssize_t i = 0; i < starts_info.size; i++) {
            if (starts_ptr[i * N] < 0) {
                throw std::invalid_argument("starts must have all non-negative values: see index [" + std::to_string(i * N) + "]");
            }
        }
        starts = starts_;
    }

    py::array_t<std::int64_t> get_stops() { return stops; }

    void set_stops(py::array stops_) {
        makeIntNative(stops_);
        stops_ = stops_.cast<py::array_t<std::int64_t>>();
        py::buffer_info stops_info = stops_.request();
        if (stops_info.ndim < 1) {
            throw std::domain_error("stops must have at least 1 dimension");
        }
        int N = stops_info.strides[0] / stops_info.itemsize;
        auto stops_ptr = (std::int64_t*)stops_info.ptr;
        for (ssize_t i = 0; i < stops_info.size; i++) {
            if (stops_ptr[i * N] < 0) {
                throw std::invalid_argument("stops must have all non-negative values: see index [" + std::to_string(i * N) + "]");
            }
        }
        stops = stops_;
    }
    
    JaggedArraySrc(py::array starts_, py::array stops_, py::object content_) {
        set_starts(starts_);
        set_stops(stops_);
        set_content(content_);
    }

    static py::array_t<std::int64_t> offsets2parents(py::array offsets) {
        makeIntNative(offsets);
        offsets = offsets.cast<py::array_t<std::int64_t>>();
        py::buffer_info offsets_info = offsets.request();
        if (offsets_info.size <= 0) {
            throw std::invalid_argument("offsets must have at least one element");
        }
        auto offsets_ptr = (std::int64_t*)offsets_info.ptr;
        int N = offsets_info.strides[0] / offsets_info.itemsize;

        ssize_t parents_length = (ssize_t)offsets_ptr[offsets_info.size - 1];
        auto parents = py::array_t<std::int64_t>(parents_length);
        py::buffer_info parents_info = parents.request();

        auto parents_ptr = (std::int64_t*)parents_info.ptr;

        ssize_t j = 0;
        ssize_t k = -1;
        for (ssize_t i = 0; i < offsets_info.size; i++) {
            while (j < (ssize_t)offsets_ptr[i * N]) {
                parents_ptr[j] = (std::int64_t)k;
                j += 1;
            }
            k += 1;
        }

        return parents;
    }

    static py::array_t<std::int64_t> counts2offsets(py::array counts) {
        makeIntNative(counts);
        counts = counts.cast<py::array_t<std::int64_t>>();
        py::buffer_info counts_info = counts.request();
        auto counts_ptr = (std::int64_t*)counts_info.ptr;
        int N = counts_info.strides[0] / counts_info.itemsize;

        ssize_t offsets_length = counts_info.size + 1;
        auto offsets = py::array_t<std::int64_t>(offsets_length);
        py::buffer_info offsets_info = offsets.request();
        auto offsets_ptr = (std::int64_t*)offsets_info.ptr;

        offsets_ptr[0] = 0;
        for (ssize_t i = 0; i < counts_info.size; i++) {
            offsets_ptr[i + 1] = offsets_ptr[i] + (std::int64_t)counts_ptr[i * N];
        }
        return offsets;
    }

    static py::array_t<std::int64_t> startsstops2parents(py::array starts, py::array stops) {
        makeIntNative(starts);
        makeIntNative(stops);
        starts = starts.cast<py::array_t<std::int64_t>>();
        stops = stops.cast<py::array_t<std::int64_t>>();
        py::buffer_info starts_info = starts.request();
        auto starts_ptr = (std::int64_t*)starts_info.ptr;
        int N_starts = starts_info.strides[0] / starts_info.itemsize;

        py::buffer_info stops_info = stops.request();
        auto stops_ptr = (std::int64_t*)stops_info.ptr;
        int N_stops = stops_info.strides[0] / stops_info.itemsize;

        ssize_t max;
        if (stops_info.size < 1) {
            max = 0;
        }
        else {
            max = (ssize_t)stops_ptr[0];
            for (ssize_t i = 1; i < stops_info.size; i++) {
                if ((ssize_t)stops_ptr[i * N_stops] > max) {
                    max = (ssize_t)stops_ptr[i * N_stops];
                }
            }
        }
        auto parents = py::array_t<std::int64_t>(max);
        py::buffer_info parents_info = parents.request();
        auto parents_ptr = (std::int64_t*)parents_info.ptr;
        for (ssize_t i = 0; i < max; i++) {
            parents_ptr[i] = -1;
        }

        for (ssize_t i = 0; i < starts_info.size; i++) {
            for (ssize_t j = (ssize_t)starts_ptr[i * N_starts]; j < (ssize_t)stops_ptr[i * N_stops]; j++) {
                parents_ptr[j] = (std::int64_t)i;
            }
        }

        return parents;
    }

    static py::tuple parents2startsstops(py::array parents, std::int64_t length = -1) {
        makeIntNative(parents);
        parents = parents.cast<py::array_t<std::int64_t>>();
        py::buffer_info parents_info = parents.request();
        auto parents_ptr = (std::int64_t*)parents_info.ptr;
        int N = parents_info.strides[0] / parents_info.itemsize;

        if (length < 0) {
            length = 0;
            for (ssize_t i = 0; i < parents_info.size; i++) {
                if (parents_ptr[i] > length) {
                    length = parents_ptr[i * N];
                }
            }
            length++;
        }

        auto starts = py::array_t<std::int64_t>((ssize_t)length);
        py::buffer_info starts_info = starts.request();
        auto starts_ptr = (std::int64_t*)starts_info.ptr;

        auto stops = py::array_t<std::int64_t>((ssize_t)length);
        py::buffer_info stops_info = stops.request();
        auto stops_ptr = (std::int64_t*)stops_info.ptr;

        for (ssize_t i = 0; i < (ssize_t)length; i++) {
            starts_ptr[i] = 0;
            stops_ptr[i] = 0;
        }

        std::int64_t last = -1;
        for (ssize_t k = 0; k < parents_info.size; k++) {
            auto thisOne = parents_ptr[k * N];
            if (last != thisOne) {
                if (last >= 0 && last < length) {
                    stops_ptr[last] = (std::int64_t)k;
                }
                if (thisOne >= 0 && thisOne < length) {
                    starts_ptr[thisOne] = (std::int64_t)k;
                }
            }
            last = thisOne;
        }

        if (last != -1) {
            stops_ptr[last] = (std::int64_t)parents_info.size;
        }

        py::list temp;
        temp.append(starts);
        temp.append(stops);
        py::tuple out(temp);
        return out;
    }

    static py::tuple uniques2offsetsparents(py::array uniques) {
        makeIntNative(uniques);
        uniques = uniques.cast<py::array_t<std::int64_t>>();
        py::buffer_info uniques_info = uniques.request();
        auto uniques_ptr = (std::int64_t*)uniques_info.ptr;
        int N = uniques_info.strides[0] / uniques_info.itemsize;

        ssize_t tempLength;
        if (uniques_info.size < 1) {
            tempLength = 0;
        }
        else {
            tempLength = uniques_info.size - 1;
        }
        auto tempArray = py::array_t<bool>(tempLength);
        py::buffer_info tempArray_info = tempArray.request();
        auto tempArray_ptr = (bool*)tempArray_info.ptr;

        ssize_t countLength = 0;
        for (ssize_t i = 0; i < uniques_info.size - 1; i++) {
            if (uniques_ptr[i * N] != uniques_ptr[(i + 1) * N]) {
                tempArray_ptr[i] = true;
                countLength++;
            }
            else {
                tempArray_ptr[i] = false;
            }
        }
        auto changes = py::array_t<std::int64_t>(countLength);
        py::buffer_info changes_info = changes.request();
        auto changes_ptr = (std::int64_t*)changes_info.ptr;
        ssize_t index = 0;
        for (ssize_t i = 0; i < tempArray_info.size; i++) {
            if (tempArray_ptr[i]) {
                changes_ptr[index++] = (std::int64_t)(i + 1);
            }
        }

        auto offsets = py::array_t<std::int64_t>(changes_info.size + 2);
        py::buffer_info offsets_info = offsets.request();
        auto offsets_ptr = (std::int64_t*)offsets_info.ptr;
        offsets_ptr[0] = 0;
        offsets_ptr[offsets_info.size - 1] = (std::int64_t)uniques_info.size;
        for (ssize_t i = 1; i < offsets_info.size - 1; i++) {
            offsets_ptr[i] = changes_ptr[i - 1];
        }

        auto parents = py::array_t<std::int64_t>(uniques_info.size);
        py::buffer_info parents_info = parents.request();
        auto parents_ptr = (std::int64_t*)parents_info.ptr;
        for (ssize_t i = 0; i < parents_info.size; i++) {
            parents_ptr[i] = 0;
        }
        for (ssize_t i = 0; i < changes_info.size; i++) {
            parents_ptr[(ssize_t)changes_ptr[i]] = 1;
        }
        for (ssize_t i = 1; i < parents_info.size; i++) {
            parents_ptr[i] += parents_ptr[i - 1];
        }

        py::list temp;
        temp.append(offsets);
        temp.append(parents);
        py::tuple out(temp);
        return out;
    }

    ContentType* __getitem__(ssize_t start, ssize_t stop) { // TODO
        throw std::invalid_argument("to be implemented later");
        return this;
    }

    ContentType* __getitem__(ssize_t index) {
        py::buffer_info starts_info = starts.request();
        py::buffer_info stops_info = stops.request();
        if (index < 0) {
            index = starts_info.size + index;
        }
        if (starts_info.size > stops_info.size) {
            throw std::out_of_range("starts must have the same or shorter length than stops");
        }
        if (index > starts_info.size || index < 0) {
            throw std::out_of_range("getitem must be in the bounds of the array");
        }
        if (starts_info.ndim != stops_info.ndim) {
            throw std::domain_error("starts and stops must have the same dimensionality");
        }
        ssize_t start = (ssize_t)((std::int64_t*)starts_info.ptr)[index];
        ssize_t stop = (ssize_t)((std::int64_t*)stops_info.ptr)[index];

        return content->__getitem__(start, stop);
    }

    std::string __str__() {
        std::string out;
        
        py::buffer_info starts_info = starts.request();
        auto starts_ptr = (std::int64_t*)starts_info.ptr;

        py::buffer_info stops_info = stops.request();
        auto stops_ptr = (std::int64_t*)stops_info.ptr;

        out.reserve(starts_info.size * 20);

        ssize_t limit = starts_info.size;
        if (limit > stops_info.size) {
            throw std::out_of_range("starts must have the same or shorter length than stops");
        }
        out.append("[");
        for (ssize_t i = 0; i < limit; i++) {
            if (i != 0) {
                out.append(" ");
            }
            out.append((__getitem__(i))->__str__());
        }
        out.append("]");
        out.shrink_to_fit();
        return out;
    }

    std::string __repr__() {
        return "<JaggedArray " + __str__() + ">";
    }

    ssize_t __len__() {
        return starts.request().size;
    }

    ContentType* __iter__() {
        iter_index = 0;
        return this;
    }

    ContentType* __next__() {
        if (iter_index >= starts.request().size) {
            throw py::stop_iteration();
        }
        return __getitem__(iter_index++);
    }
};

PYBIND11_MODULE(_jagged, m) {
    py::class_<ContentType>(m, "ContentType");
    py::class_<NumpyArray>(m, "NumpyArray")
        .def(py::init<py::array>())
        .def("__str__", &NumpyArray::__str__)
        .def("__repr__", &NumpyArray::__repr__)
        .def("__len__", &NumpyArray::__len__)
        .def("__getitem__", (ContentType* (NumpyArray::*)(ssize_t)) &NumpyArray::__getitem__)
        .def("__getitem__", (ContentType* (NumpyArray::*)(ssize_t, ssize_t)) &NumpyArray::__getitem__);
    py::class_<AwkwardArray>(m, "AwkwardArray");
    py::class_<JaggedArraySrc>(m, "JaggedArraySrc")
        .def(py::init<py::array, py::array, py::object>())
        .def_property("starts", &JaggedArraySrc::get_starts, &JaggedArraySrc::set_starts)
        .def_property("stops", &JaggedArraySrc::get_stops, &JaggedArraySrc::set_stops)
        .def_property("content", &JaggedArraySrc::get_content, &JaggedArraySrc::set_content)
        .def_static("offsets2parents", &JaggedArraySrc::offsets2parents)
        .def_static("counts2offsets", &JaggedArraySrc::counts2offsets)
        .def_static("startsstops2parents", &JaggedArraySrc::startsstops2parents)
        .def_static("parents2startsstops", &JaggedArraySrc::parents2startsstops)
        .def_static("uniques2offsetsparents", &JaggedArraySrc::uniques2offsetsparents)
        .def("__getitem__", (ContentType* (JaggedArraySrc::*)(ssize_t)) &JaggedArraySrc::__getitem__)
        .def("__getitem__", (ContentType* (JaggedArraySrc::*)(ssize_t, ssize_t)) &JaggedArraySrc::__getitem__)
        .def("__str__", &JaggedArraySrc::__str__)
        .def("__repr__", &JaggedArraySrc::__repr__)
        .def("__len__", &JaggedArraySrc::__len__)
        .def("__iter__", &JaggedArraySrc::__iter__)
        .def("__next__", &JaggedArraySrc::__next__);
}
