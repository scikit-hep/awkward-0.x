/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
TODO:
= Expand array types
    - std::string
    - py::object
    - zero-terminated bytes
    - raw data
= Add more getitem functions
= Deal with more array characteristics
    - Multidimensional arrays
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
#include <stdexcept>
#include <sstream>
#include "util.h"
#include "any.h"
#include "numpytypes.h"

namespace py = pybind11;

class JaggedArray : public AwkwardArray {
public:
    py::array_t<std::int64_t> starts,
                              stops;
    AnyArray*                 content;

    py::object unwrap() {
        return py::cast(this);
    }

    AnyArray* get_content() { return content; }

    py::object python_get_content() {
        return content->unwrap();
    }

    void set_content(AnyArray* content_) {
        content = content_;
    }

    void python_set_content(py::object content_) {
        try {
            set_content(content_.cast<JaggedArray*>());
            return;
        }
        catch (py::cast_error e) { }
        try {
            set_content(getNumpyArray_t(content_.cast<py::array>()));
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

    bool check_validity() {
        py::buffer_info starts_info = starts.request();
        py::buffer_info stops_info = stops.request();
        if (starts_info.size > stops_info.size) {
            throw std::invalid_argument("starts must have the same (or shorter) length than stops");
        }
        if (starts_info.ndim != stops_info.ndim) {
            throw std::domain_error("starts and stops must have the same dimensionality");
        }
        int N_starts = starts_info.strides[0] / starts_info.itemsize;
        int N_stops = stops_info.strides[0] / stops_info.itemsize;
        std::int64_t starts_max = 0;
        std::int64_t stops_max = 0;
        auto starts_ptr = (std::int64_t*)starts_info.ptr;
        auto stops_ptr = (std::int64_t*)stops_info.ptr;
        for (ssize_t i = 0; i < starts_info.size; i++) {
            if (stops_ptr[i * N_stops] < starts_ptr[i * N_starts]) {
                throw std::invalid_argument("stops must be greater than or equal to starts");
            }
            if (starts_ptr[i * N_starts] > starts_max) {
                starts_max = starts_ptr[i * N_starts];
            }
            if (stops_ptr[i * N_stops] > stops_max) {
                stops_max = stops_ptr[i * N_stops];
            }
        }
        if (starts_info.size > 0) {
            if (starts_max >= content->len()) {
                throw std::invalid_argument("The maximum of starts for non-empty elements must be less than the length of content");
            }
            if (stops_max > content->len()) {
                throw std::invalid_argument("The maximum of stops for non-empty elements must be less than or equal to the length of content");
            }
        }
        return true;
    }

    JaggedArray(py::array starts_, py::array stops_, py::object content_) {
        set_starts(starts_);
        set_stops(stops_);
        python_set_content(content_);
        check_validity();
    }

    JaggedArray(py::array starts_, py::array stops_, AnyArray* content_) {
        set_starts(starts_);
        set_stops(stops_);
        set_content(content_);
        check_validity();
    }

    static JaggedArray* fromoffsets(py::array offsets, AnyArray* content_) {
        makeIntNative(offsets);
        py::array_t<std::int64_t> temp = offsets.cast<py::array_t<std::int64_t>>();
        ssize_t length = temp.request().size;
        if (length < 1) {
            throw std::invalid_argument("offsets must have at least one element");
        }
        if (temp.request().ndim > 1) {
            throw std::domain_error("offsets must be one-dimensional");
        }
        return new JaggedArray(
            slice_numpy(temp, 0, length - 1),
            slice_numpy(temp, 1, length - 1),
            content_
        );
    }

    static JaggedArray* python_fromoffsets(py::array offsets, py::object content_) {
        try {
            return fromoffsets(offsets, content_.cast<JaggedArray*>());
        }
        catch (py::cast_error e) { }
        try {
            return fromoffsets(offsets, getNumpyArray_t(content_.cast<py::array>()));
        }
        catch (py::cast_error e) {
            throw std::invalid_argument("Invalid type for JaggedArray.content");
        }
    }

    static JaggedArray* fromcounts(py::array counts, AnyArray* content_) {
        return fromoffsets(counts2offsets(counts), content_);
    }

    static JaggedArray* python_fromcounts(py::array counts, py::object content_) {
        try {
            return fromcounts(counts, content_.cast<JaggedArray*>());
        }
        catch (py::cast_error e) { }
        try {
            return fromcounts(counts, getNumpyArray_t(content_.cast<py::array>()));
        }
        catch (py::cast_error e) {
            throw std::invalid_argument("Invalid type for JaggedArray.content");
        }
    }

    static JaggedArray* fromparents(py::array parents, AnyArray* content_, ssize_t length = -1) {
        if (parents.request().ndim != 1 || parents.request().size != content_->len()) {
            throw std::invalid_argument("parents array must be one-dimensional with the same length as content");
        }
        auto startsstops = parents2startsstops(parents, length);
        return new JaggedArray(startsstops[0], startsstops[1], content_);
    }

    static JaggedArray* python_fromparents(py::array parents, py::object content_, ssize_t length = -1) {
        try {
            return fromparents(parents, content_.cast<JaggedArray*>(), length);
        }
        catch (py::cast_error e) { }
        try {
            return fromparents(parents, getNumpyArray_t(content_.cast<py::array>()), length);
        }
        catch (py::cast_error e) {
            throw std::invalid_argument("Invalid type for JaggedArray.content");
        }
    }

    static JaggedArray* fromuniques(py::array uniques, AnyArray* content_) {
        if (uniques.request().ndim != 1 || uniques.request().size != content_->len()) {
            throw std::invalid_argument("uniques array must be one-dimensional with the same length as content");
        }
        auto offsetsparents = uniques2offsetsparents(uniques);
        return fromoffsets(offsetsparents[0], content_);
    }

    static JaggedArray* python_fromuniques(py::array uniques, py::object content_) {
        try {
            return fromuniques(uniques, content_.cast<JaggedArray*>());
        }
        catch (py::cast_error e) { }
        try {
            return fromuniques(uniques, getNumpyArray_t(content_.cast<py::array>()));
        }
        catch (py::cast_error e) {
            throw std::invalid_argument("Invalid type for JaggedArray.content");
        }
    }

    static JaggedArray* fromjagged(JaggedArray* jagged) {
        return new JaggedArray(jagged->get_starts(), jagged->get_stops(), jagged->get_content());
    }

    JaggedArray* copy() {
        return new JaggedArray(starts, stops, content);
    }

    AnyArray* deepcopy() {
        return new JaggedArray(
            pyarray_deepcopy(starts),
            pyarray_deepcopy(stops),
            content->deepcopy()
        );
    }

    JaggedArray* python_deepcopy() {
        return (JaggedArray*)deepcopy();
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

    ssize_t len() {
        return starts.request().size;
    }

    AnyArray* getitem(ssize_t start, ssize_t length, ssize_t step = 1) {
        if (step == 0) {
            throw std::invalid_argument("slice step cannot be 0");
        }
        if (length < 0) {
            throw std::invalid_argument("slice length cannot be less than 0");
        }
        if (start < 0 || start >= len() || start + (length * step) > len() || start + (length * step) < -1) {
            throw std::out_of_range("getitem must be in the bounds of the array.");
        }
        auto newStarts = py::array_t<std::int64_t>(length);
        py::buffer_info newStarts_info = newStarts.request();
        auto newStarts_ptr = (std::int64_t*)newStarts_info.ptr;

        py::buffer_info starts_info = starts.request();
        auto starts_ptr = (std::int64_t*)starts_info.ptr;
        int N_starts = starts_info.strides[0] / starts_info.itemsize;

        auto newStops = py::array_t<std::int64_t>(length);
        py::buffer_info newStops_info = newStops.request();
        auto newStops_ptr = (std::int64_t*)newStops_info.ptr;

        py::buffer_info stops_info = stops.request();
        auto stops_ptr = (std::int64_t*)stops_info.ptr;
        int N_stops = stops_info.strides[0] / stops_info.itemsize;

        ssize_t newIndex = 0;
        for (ssize_t i = 0; i < length; i++) {
            newStarts_ptr[newIndex] = starts_ptr[start + (i * step * N_starts)];
            newStops_ptr[newIndex++] = stops_ptr[start + (i * step * N_stops)];
        }

        return new JaggedArray(newStarts, newStops, content);
    }

    py::object python_getitem(ssize_t start, ssize_t stop, ssize_t step) {
        if (step == 0) {
            throw std::invalid_argument("slice step cannot be 0");
        }
        ssize_t length = len();
        if (start < 0) {
            start += length;
        }
        if (stop < 0) {
            stop += length;
        }
        if (step > 0) {
            if (stop > length) {
                stop = length;
            }
            if (start >= stop) {
                return getitem(length - 1, 0, step)->unwrap();
            }
        }
        else {
            if (stop < -1) {
                stop = -1;
            }
            if (start <= stop) {
                return getitem(0, 0, step)->unwrap();
            }
        }
        return getitem(start, (stop + step - start) / step, step)->unwrap();
    }

    AnyArray* getitem(ssize_t index) {
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

        return content->getitem(start, stop - start);
    }

    py::object python_getitem(ssize_t index) {
        return getitem(index)->unwrap();
    }

    std::string str() {
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
            out.append((getitem(i))->str());
        }
        out.append("]");
        out.shrink_to_fit();
        return out;
    }

    std::string repr() {
        std::stringstream stream;
        stream << std::hex << (long)this;
        return "<JaggedArray " + str() + " at 0x" + stream.str() + ">";
    }

    class JaggedArrayIterator {
    private:
        JaggedArray* thisArray;
        ssize_t      iter_index;

    public:
        JaggedArrayIterator(JaggedArray* thisArray_) {
            iter_index = 0;
            thisArray = thisArray_;
        }

        JaggedArrayIterator* iter() {
            return this;
        }

        py::object next() {
            if (iter_index >= thisArray->len()) {
                throw py::stop_iteration();
            }
            return thisArray->getitem(iter_index++)->unwrap();
        }
    };

    JaggedArrayIterator* iter() {
        return new JaggedArrayIterator(this);
    }
};
