#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <stdexcept>
#include "awk_util.h"

namespace py = pybind11;

class ContentType {
public:
    //virtual std::string __str__() { return ""; }
    //virtual std::string __repr__() { return ""; }
};

class NumpyArray : public ContentType {
private:
    py::array thisArray;
public:
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
            NumpyArray temp = NumpyArray(content_.cast<py::array>());
            content = &temp;
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

    template <typename T>
    py::array __getitem__(T index) { // limited to single-int arguments, with 1d non-stride content array
        /*py::buffer_info starts_info = starts.request();
        py::buffer_info stops_info = stops.request();
        if (index < 0) {
            index = starts_info.size + index;
        }
        if (starts_info.size > stops_info.size) {
            throw std::out_of_range("starts must have the same or shorter length than stops");
        }
        if ((ssize_t)index > starts_info.size || index < 0) {
            throw std::out_of_range("index must specify a location within the JaggedArray");
        }
        if (starts_info.ndim != stops_info.ndim) {
            throw std::domain_error("starts and stops must have the same dimensionality");
        }
        ssize_t start = (ssize_t)((std::int64_t*)starts_info.ptr)[index];
        ssize_t stop = (ssize_t)((std::int64_t*)stops_info.ptr)[index];
        if (content_type == 'a') {
            py::buffer_info content_info = content_array.request();
            auto content_ptr = (std::int64_t*)content_info.ptr;
            if (content_info.ndim == 1 && content_info.strides[0] == 8) {
                if (start >= content_info.size || start < 0 || stop > content_info.size || stop < 0) {
                    throw std::out_of_range("starts and stops are not within the bounds of content");
                }
                if (stop >= start) {
                    auto out = py::array_t<std::int64_t>(stop - start);
                    py::buffer_info out_info = out.request();
                    auto out_ptr = (std::int64_t*)out_info.ptr;

                    ssize_t here = 0;
                    for (ssize_t i = start; i < stop; i++) {
                        out_ptr[here++] = content_ptr[i];
                    }
                    return out;
                }
                throw std::out_of_range("stops must be greater than or equal to starts");
            }
        }*/
        auto out = py::array_t<std::int64_t>(0);
        return out;
    }

    std::string __str__() {
        /*std::string out = "";

        py::buffer_info starts_info = starts.request();
        auto starts_ptr = (std::int64_t*)starts_info.ptr;

        py::buffer_info stops_info = stops.request();
        auto stops_ptr = (std::int64_t*)stops_info.ptr;


            py::buffer_info array_info = content_array.request();
            auto array_ptr = (std::int64_t*)array_info.ptr;

            if (array_info.ndim == 1 && array_info.strides[0] / array_info.itemsize == 1) {
                if (starts_info.size > stops_info.size) {
                    throw std::out_of_range("starts must be the same or shorter length than stops");
                }

                ssize_t limit = starts_info.size;
                for (ssize_t i = 0; i < limit; i++) {
                    if (i != 0) {
                        out = out + " ";
                    }
                    out = out + "[";

                    ssize_t end = (ssize_t)stops_ptr[i];
                    if ((ssize_t)starts_ptr[i] > end) {
                        throw std::out_of_range("stops must be greater than or equal to starts");
                    }

                    for (ssize_t j = (ssize_t)starts_ptr[i]; j < end; j++) {
                        if (j != (ssize_t)starts_ptr[i]) {
                            out = out + " ";
                        }
                        out = out + std::to_string(array_ptr[j]);
                    }
                    out = out + "]";
                }
                return "[" + out + "]";
            }
            */
        return "-Error: print function is not yet implemented for this type-";
    }

    std::string __repr__() { // limited functionality
        return "<JaggedArray " + __str__() + ">";
    }

    ssize_t __len__() {
        return starts.request().size;
    }

    JaggedArraySrc* __iter__() {
        iter_index = 0;
        return this;
    }

    py::array_t<int64_t> __next__() { // very limited, like getitem and repr
        if (iter_index >= starts.request().size) {
            throw py::stop_iteration();
        }
        return __getitem__(iter_index++);
    }
};

PYBIND11_MODULE(_jagged, m) {
    py::class_<ContentType>(m, "ContentType");
    py::class_<NumpyArray>(m, "NumpyArray")
        .def(py::init<py::array>());
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
        .def("__getitem__", &JaggedArraySrc::__getitem__<py::array>)
        .def("__getitem__", &JaggedArraySrc::__getitem__<std::int64_t>)
        .def("__str__", &JaggedArraySrc::__str__)
        .def("__repr__", &JaggedArraySrc::__repr__)
        .def("__len__", &JaggedArraySrc::__len__)
        .def("__iter__", &JaggedArraySrc::__iter__)
        .def("__next__", &JaggedArraySrc::__next__);
}
