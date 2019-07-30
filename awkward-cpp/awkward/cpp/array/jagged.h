#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cinttypes>
#include <stdexcept>
#include <sstream>
#include "util.h"
#include "any.h"
#include "numpytypes.h"
#include "cpu_methods.h"
#include "cpu_pybind11.h"

#include <stdio.h> // for debugging purposes

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
        makeIntNative_CPU(starts_);
        starts_ = starts_.cast<py::array_t<std::int64_t>>();
        py::buffer_info starts_info = starts_.request();
        struct c_array starts_struct = py2c(&starts_info);
        if (starts_info.ndim < 1) {
            throw std::domain_error("starts must have at least 1 dimension");
        }
        if (!checkNonNegative_CPU(&starts_struct)) {
            throw std::invalid_argument("starts must have all non-negative values");
        }
        starts = starts_;
    }

    void python_set_starts(py::object input) {
        py::array starts_ = input.cast<py::array>();
        set_starts(starts_);
    }

    py::array_t<std::int64_t> get_stops() { return stops; }

    void set_stops(py::array stops_) {
        makeIntNative_CPU(stops_);
        stops_ = stops_.cast<py::array_t<std::int64_t>>();
        py::buffer_info stops_info = stops_.request();
        struct c_array stops_struct = py2c(&stops_info);
        if (stops_info.ndim < 1) {
            throw std::domain_error("stops must have at least 1 dimension");
        }
        if (!checkNonNegative_CPU(&stops_struct)) {
            throw std::invalid_argument("stops must have all non-negative values");
        }
        stops = stops_;
    }

    void python_set_stops(py::object input) {
        py::array stops_ = input.cast<py::array>();
        set_stops(stops_);
    }

    bool check_validity() {
        py::buffer_info starts_info = starts.request();
        struct c_array starts_struct = py2c(&starts_info);
        py::buffer_info stops_info = stops.request();
        struct c_array stops_struct = py2c(&stops_info);
        if (starts_info.size > stops_info.size) {
            throw std::invalid_argument("starts must have the same (or shorter) length than stops");
        }
        if (starts_info.ndim != stops_info.ndim) {
            throw std::domain_error("starts and stops must have the same dimensionality");
        }
        std::int64_t starts_max = 0;
        getMax_CPU(starts, &starts_max);
        std::int64_t stops_max = 0;
        getMax_CPU(stops, &stops_max);
        std::string comparison = "<=";
        if (!compare_CPU(&starts_struct, &stops_struct, comparison.c_str())) {
            throw std::invalid_argument("starts must be less than or equal to stops");
        }
        if (starts_info.size > 0) {
            if (starts_max > content->len()) {
                throw std::invalid_argument("The maximum of starts for non-empty elements must be less than or equal to the length of content");
            }
            if (stops_max > content->len()) {
                throw std::invalid_argument("The maximum of stops for non-empty elements must be less than or equal to the length of content");
            }
        }
        return true;
    }

    JaggedArray(py::object starts_, py::object stops_, py::object content_) {
        python_set_starts(starts_);
        python_set_stops(stops_);
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
        makeIntNative_CPU(offsets);
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

    static JaggedArray* python_fromoffsets(py::object input, py::object content_) {
        py::array offsets = input.cast<py::array>();
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

    static JaggedArray* python_fromcounts(py::object input, py::object content_) {
        py::array counts = input.cast<py::array>();
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

    static AnyArray* fromiter_helper(py::tuple input) {
        if (input.size() == 0) {
            return getNumpyArray_t(py::array_t<std::int32_t>(0));
        }
        try {
            input[0].cast<py::tuple>();
            return fromiter(input);
        }
        catch (std::exception e) {
            py::array out = input.cast<py::array>();
            return getNumpyArray_t(out);
        }
    }

    static JaggedArray* fromiter(py::object input) {
        py::tuple iter = input.cast<py::tuple>();
        auto counts = py::array_t<std::int64_t>(iter.size());
        auto counts_ptr = (std::int64_t*)counts.request().ptr;

        py::list contentList;

        if (iter.size() == 0) {
            return fromcounts(counts, getNumpyArray_t(py::array_t<std::int32_t>(0)));
        }
        for (size_t i = 0; i < iter.size(); i++) {
            py::tuple thisIter;
            try {
                thisIter = iter[i].cast<py::tuple>();
            }
            catch (std::exception e) {
                throw std::invalid_argument("jagged iterable must contain only iterables to make a jagged array");
            }
            counts_ptr[i] = (std::int64_t)thisIter.size();
            for (size_t i = 0; i < thisIter.size(); i++) {
                contentList.append(thisIter[i]);
            }
        }
        auto content_out = py::tuple(contentList);
        return fromcounts(counts, fromiter_helper(content_out));
    }

    static JaggedArray* fromparents(py::array parents, AnyArray* content_, ssize_t length = -1) {
        if (parents.request().ndim != 1 || parents.request().size != content_->len()) {
            throw std::invalid_argument("parents array must be one-dimensional with the same length as content");
        }
        auto startsstops = parents2startsstops(parents, length);
        return new JaggedArray(startsstops[0], startsstops[1], content_);
    }

    static JaggedArray* python_fromparents(py::object input, py::object content_, ssize_t length = -1) {
        py::array parents = input.cast<py::array>();
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

    static JaggedArray* python_fromuniques(py::object input, py::object content_) {
        py::array uniques = input.cast<py::array>();
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
        makeIntNative_CPU(offsets);
        offsets = offsets.cast<py::array_t<std::int64_t>>();
        py::buffer_info offsets_info = offsets.request();
        struct c_array offsets_struct = py2c(&offsets_info);
        if (offsets_info.size <= 0) {
            throw std::invalid_argument("offsets must have at least one element");
        }
        auto offsets_ptr = (std::int64_t*)offsets_info.ptr;
        int N = offsets_info.strides[0] / offsets_info.itemsize;

        ssize_t parents_length = (ssize_t)offsets_ptr[(offsets_info.size - 1) * N];
        auto parents = py::array_t<std::int64_t>(parents_length);
        py::buffer_info parents_info = parents.request();
        struct c_array parents_struct = py2c(&parents_info);

        if (!offsets2parents_CPU(&offsets_struct, &parents_struct)) {
            throw std::invalid_argument("Error in cpu_methods.h::offsets2parents_CPU");
        }
        return parents;
    }

    static py::array_t<std::int64_t> python_offsets2parents(py::object offsetsIter) {
        py::array offsets = offsetsIter.cast<py::array>();
        return offsets2parents(offsets);
    }

    static py::array_t<std::int64_t> counts2offsets(py::array counts) {
        makeIntNative_CPU(counts);
        counts = counts.cast<py::array_t<std::int64_t>>();
        py::buffer_info counts_info = counts.request();
        struct c_array counts_struct = py2c(&counts_info);
        auto offsets = py::array_t<std::int64_t>(counts_info.size + 1);
        py::buffer_info offsets_info = offsets.request();
        struct c_array offsets_struct = py2c(&offsets_info);

        if (!counts2offsets_CPU(&counts_struct, &offsets_struct)) {
            throw std::invalid_argument("Error in cpu_methods.h::counts2offsets_CPU");
        }
        return offsets;
    }

    static py::array_t<std::int64_t> python_counts2offsets(py::object countsIter) {
        py::array counts = countsIter.cast<py::array>();
        return counts2offsets(counts);
    }

    static py::array_t<std::int64_t> startsstops2parents(py::array starts_, py::array stops_) {
        makeIntNative_CPU(starts_);
        makeIntNative_CPU(stops_);
        starts_ = starts_.cast<py::array_t<std::int64_t>>();
        py::buffer_info starts_info = starts_.request();
        struct c_array starts_struct = py2c(&starts_info);
        stops_ = stops_.cast<py::array_t<std::int64_t>>();
        py::buffer_info stops_info = stops_.request();
        struct c_array stops_struct = py2c(&stops_info);

        std::int64_t max = 0;
        getMax_CPU(stops_, &max);
        auto parents = py::array_t<std::int64_t>((ssize_t)max);
        py::buffer_info parents_info = parents.request();
        struct c_array parents_struct = py2c(&parents_info);

        if (!startsstops2parents_CPU(&starts_struct, &stops_struct, &parents_struct)) {
            throw std::invalid_argument("Error in cpu_methods.h::startsstops2parents_CPU");
        }
        return parents;
    }

    static py::array_t<std::int64_t> python_startsstops2parents(py::object startsIter, py::object stopsIter) {
        py::array starts_ = startsIter.cast<py::array>();
        py::array stops_ = stopsIter.cast<py::array>();
        return startsstops2parents(starts_, stops_);
    }

    static py::tuple parents2startsstops(py::array parents, std::int64_t length = -1) {
        makeIntNative_CPU(parents);
        parents = parents.cast<py::array_t<std::int64_t>>();
        py::buffer_info parents_info = parents.request();
        struct c_array parents_struct = py2c(&parents_info);

        if (length < 0) {
            length = 0;
            getMax_CPU(parents, &length);
            length++;
        }
        auto starts_ = py::array_t<std::int64_t>((ssize_t)length);
        py::buffer_info starts_info = starts_.request();
        struct c_array starts_struct = py2c(&starts_info);
        auto stops_ = py::array_t<std::int64_t>((ssize_t)length);
        py::buffer_info stops_info = stops_.request();
        struct c_array stops_struct = py2c(&stops_info);

        if (!parents2startsstops_CPU(&parents_struct, &starts_struct, &stops_struct)) {
            throw std::invalid_argument("Error in cpu_methods.h::parents2startsstops_CPU");
        }
        py::list temp;
        temp.append(starts_);
        temp.append(stops_);
        py::tuple out(temp);
        return out;
    }

    static py::tuple python_parents2startsstops(py::object parentsIter, std::int64_t length = -1) {
        py::array parents = parentsIter.cast<py::array>();
        return parents2startsstops(parents, length);
    }

    static py::tuple uniques2offsetsparents(py::array uniques) {
        makeIntNative_CPU(uniques);
        uniques = uniques.cast<py::array_t<std::int64_t>>();
        py::buffer_info uniques_info = uniques.request();
        struct c_array uniques_struct = py2c(&uniques_info);

        ssize_t tempLength = 0;
        if (uniques_info.size > 0) {
            tempLength = uniques_info.size - 1;
        }

        auto tempArray = py::array_t<std::int8_t>(tempLength);
        py::buffer_info temp_info = tempArray.request();
        struct c_array temp_struct = py2c(&temp_info);
        ssize_t countLength = 0;
        if (!uniques2offsetsparents_generateTemparray_CPU(&uniques_struct, &temp_struct, &countLength)) {
            throw std::invalid_argument("Error in cpu_methods.h::uniques2offsetsparents_generateTempArray_CPU");
        }
        auto offsets = py::array_t<std::int64_t>(countLength + 2);
        py::buffer_info offsets_info = offsets.request();
        struct c_array offsets_struct = py2c(&offsets_info);
        auto parents = py::array_t<std::int64_t>(uniques_info.size);
        py::buffer_info parents_info = parents.request();
        struct c_array parents_struct = py2c(&parents_info);
        if (!uniques2offsetsparents_CPU(countLength, &temp_struct, &offsets_struct, &parents_struct)) {
            throw std::invalid_argument("Error in cpu_methods.h::uniques2offsetsparents_CPU");
        }

        py::list temp;
        temp.append(offsets);
        temp.append(parents);
        py::tuple out(temp);
        return out;
    }

    static py::tuple python_uniques2offsetsparents(py::object uniquesIter) {
        py::array uniques = uniquesIter.cast<py::array>();
        return uniques2offsetsparents(uniques);
    }

    ssize_t len() {
        return starts.request().size;
    }

    AnyArray* getitem(ssize_t start, ssize_t length, ssize_t step = 1) {
        return new JaggedArray(
            slice_numpy(starts, start, length, step),
            slice_numpy(stops, start, length, step),
            content
        );
    }

    py::object python_getitem(py::slice input) {
        size_t start, stop, step, slicelength;
        if (!input.compute(len(), &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
        }
        return getitem((ssize_t)start, (ssize_t)slicelength, (ssize_t)step)->unwrap();
    }

    AnyOutput* getitem(ssize_t index) {
        py::buffer_info starts_info = starts.request();
        py::buffer_info stops_info = stops.request();
        if (starts_info.size > stops_info.size) {
            throw std::out_of_range("starts must have the same or shorter length than stops");
        }
        if (index > starts_info.size || index < 0) {
            throw std::out_of_range("getitem must be in the bounds of the array");
        }
        if (starts_info.ndim != stops_info.ndim) {
            throw std::domain_error("starts and stops must have the same dimensionality");
        }
        int N_starts = starts_info.strides[0] / starts_info.itemsize;
        int N_stops = stops_info.strides[0] / stops_info.itemsize;
        ssize_t start = (ssize_t)((std::int64_t*)starts_info.ptr)[index * N_starts];
        ssize_t stop = (ssize_t)((std::int64_t*)stops_info.ptr)[index * N_stops];

        return content->getitem(start, stop - start);
    }

    py::object python_getitem(ssize_t index) {
        if (index < 0) {
            index += starts.request().size;
        }
        return getitem(index)->unwrap();
    }

    AnyArray* boolarray_getitem(py::array input) {
        ssize_t length = input.request().size;
        if (length != len()) {
            throw std::invalid_argument("bool array length must be equal to jagged array length");
        }
        auto array_ptr = (bool*)input.request().ptr;

        py::list tempStarts;
        py::list tempStops;

        py::buffer_info starts_info = starts.request();
        auto starts_ptr = (std::int64_t*)starts_info.ptr;
        int N_starts = starts_info.strides[0] / starts_info.itemsize;

        py::buffer_info stops_info = stops.request();
        auto stops_ptr = (std::int64_t*)stops_info.ptr;
        int N_stops = stops_info.strides[0] / stops_info.itemsize;

        for (ssize_t i = 0; i < length; i++) {
            if (array_ptr[i]) {
                tempStarts.append(starts_ptr[i * N_starts]);
                tempStops.append(stops_ptr[i * N_stops]);
            }
        }
        py::array_t<std::int64_t> outStarts = tempStarts.cast<py::array_t<std::int64_t>>();
        py::array_t<std::int64_t> outStops = tempStops.cast<py::array_t<std::int64_t>>();
        return new JaggedArray(outStarts, outStops, content);
    }

    AnyArray* intarray_getitem(py::array input) {
        makeIntNative_CPU(input);
        input = input.cast<py::array_t<std::int64_t>>();
        py::buffer_info array_info = input.request();
        auto array_ptr = (std::int64_t*)array_info.ptr;

        auto newStarts = py::array_t<std::int64_t>(array_info.size);
        auto newStarts_ptr = (std::int64_t*)newStarts.request().ptr;

        py::buffer_info starts_info = starts.request();
        auto starts_ptr = (std::int64_t*)starts_info.ptr;
        int N_starts = starts_info.strides[0] / starts_info.itemsize;

        auto newStops = py::array_t<std::int64_t>(array_info.size);
        auto newStops_ptr = (std::int64_t*)newStops.request().ptr;

        py::buffer_info stops_info = stops.request();
        auto stops_ptr = (std::int64_t*)stops_info.ptr;
        int N_stops = stops_info.strides[0] / stops_info.itemsize;

        for (ssize_t i = 0; i < array_info.size; i++) {
            std::int64_t here = array_ptr[i];
            if (here < 0 || here >= len()) {
                throw std::invalid_argument("int array indices must be within the bounds of the jagged array");
            }
            newStarts_ptr[i] = starts_ptr[N_starts * here];
            newStops_ptr[i] = stops_ptr[N_stops * here];
        }
        return new JaggedArray(newStarts, newStops, content);
    }

    AnyArray* getitem(py::array input) {
        if (input.request().format.find("?") != std::string::npos) {
            return boolarray_getitem(input);
        }
        return intarray_getitem(input);
    }

    py::object python_getitem(py::array input) {
        return getitem(input)->unwrap();
    }

    static AnyOutput* fromiter_any(py::object input) {
        py::tuple iter = input.cast<py::tuple>();
        try {
            iter[0];
        }
        catch (std::exception e) {
            py::list temp;
            temp.append(input);
            py::array tempArray = temp.cast<py::array>();
            NumpyArray* tempNumpy = getNumpyArray_t(tempArray);
            AnyOutput* out = tempNumpy->getitem(0);
            return out;
        }
        if (iter.size() > 0) {
            try {
                py::tuple inner = iter[0].cast<py::tuple>();
                inner[0];
                JaggedArray* out = fromiter(input);
                return out;
            }
            catch (std::exception e) { }
        }
        py::array out = input.cast<py::array>();
        return getNumpyArray_t(out);
    }

    py::object getitem_tuple(py::tuple input, ssize_t index = 0, ssize_t select_index = -1) {
        if (index < 0 || index >= (ssize_t)input.size()) { // end case
            return unwrap();
        }
        try { // int input
            ssize_t here = input[index].cast<ssize_t>();
            return getitem(here)->getitem_tuple(input, index + 1, select_index);
        }
        catch (py::cast_error e) { }
        try { // array input --> goes past the catch block
            py::tuple check = input[index].cast<py::tuple>();
            check[0];
        }
        catch (std::exception e) {
            try { // slice input
                py::slice here = input[index].cast<py::slice>();
                size_t start, stop, step, slicelength;
                if (!here.compute(len(), &start, &stop, &step, &slicelength)) {
                    throw py::error_already_set();
                }
                AnyArray* fromHere = getitem((ssize_t)start, (ssize_t)slicelength, (ssize_t)step);
                /*if (index + 1 < (ssize_t)input.size()) {
                    bool nextIsArray = false;
                    try {
                        py::tuple check = input[index + 1].cast<py::tuple>();
                        check[0];
                        nextIsArray = True;
                    }
                    catch (std::exception) { }
                    if (nextIsArray) {
                        py::list temp;
                        for (ssize_t i = 0; i < fromHere->len(); i++) {

                        }
                    }
                }*/
                py::list temp;
                for (ssize_t i = 0; i < fromHere->len(); i++) {
                    temp.append(fromHere->getitem(i)->getitem_tuple(input, index + 1, select_index));
                }
                return fromiter_any(temp)->unwrap();
            }
            catch (py::cast_error e) {
                throw std::invalid_argument("argument index for __getitem__(tuple) not recognized");
            }
        } // continued array input
        py::array here = input[index].cast<py::array>();
        if (select_index < 0) {
            AnyArray* fromHere = getitem(here);
            if (fromHere->len() == 1) {
                py::list temp;
                temp.append(fromHere->getitem(0)->getitem_tuple(input, index + 1, select_index));
                return fromiter_any(temp)->unwrap();
            }
            py::list temp;
            for (ssize_t i = 0; i < fromHere->len(); i++) {
                temp.append(fromHere->getitem(i)->getitem_tuple(input, index + 1, i));
            }
            return fromiter_any(temp)->unwrap();
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
            py::list out;
            out.append(getitem(((ssize_t*)indices.request().ptr)[0])->getitem_tuple(input, index + 1, select_index));
            return fromiter_any(out)->unwrap();
        }
        if (select_index > indices.request().size) {
            throw std::domain_error("Error: selection index exceeded selection length");
        }
        return getitem(((ssize_t*)indices.request().ptr)[select_index])->getitem_tuple(input, index + 1, select_index);
    }

    py::object python_getitem(py::tuple input) {
        return getitem_tuple(input);
    }

    AnyOutput* getitem(py::tuple input) {
        return fromiter_any(getitem_tuple(input));
    }

    py::object booljagged_getitem(JaggedArray* input) {
        if (input->len() != len()) {
            throw std::domain_error("bool array shape does not match array");
        }
        JaggedArray* inside = dynamic_cast<JaggedArray*>(content);
        JaggedArray* input_inside = dynamic_cast<JaggedArray*>(input->get_content());
        if (input_inside != 0) {
            if (inside != 0) {
                py::list out;
                for (ssize_t i = 0; i < len(); i++) {
                    inside = dynamic_cast<JaggedArray*>(getitem(i));
                    input_inside = dynamic_cast<JaggedArray*>(input->getitem(i));
                    out.append(inside->booljagged_getitem(input_inside));
                }
                return out;
            }
            throw std::domain_error("cannot have doubly jagged mask with singly jagged array");
        }
        AnyArray* this_array;
        NumpyArray* input_array;
        py::array input_unwrap;
        py::list out;
        for (ssize_t i = 0; i < len(); i++) {
            this_array = dynamic_cast<AnyArray*>(getitem(i));
            input_array = dynamic_cast<NumpyArray*>(input->getitem(i));
            input_unwrap = input_array->unwrap().cast<py::array>();
            out.append(this_array->boolarray_getitem(input_unwrap)->unwrap());
        }
        return out;
    }

    py::object intjagged_getitem(JaggedArray* input) {
        JaggedArray* inside = dynamic_cast<JaggedArray*>(content);
        JaggedArray* input_inside = dynamic_cast<JaggedArray*>(input->get_content());
        if (input_inside != 0) {
            if (inside != 0) {
                if (input->len() != len()) {
                    throw std::domain_error("int array shape does not match array");
                }
                py::list out;
                for (ssize_t i = 0; i < input->len(); i++) {
                    inside = dynamic_cast<JaggedArray*>(getitem(i));
                    input_inside = dynamic_cast<JaggedArray*>(input->getitem(i));
                    out.append(inside->intjagged_getitem(input_inside));
                }
                return out;
            }
            throw std::domain_error("cannot have doubly jagged int array with single jagged array");
        }
        AnyArray* this_array;
        NumpyArray* input_array;
        py::array input_unwrap;
        py::list out;
        for (ssize_t i = 0; i < len(); i++) {
            this_array = dynamic_cast<AnyArray*>(getitem(i));
            input_array = dynamic_cast<NumpyArray*>(input->getitem(i));
            input_unwrap = input_array->unwrap().cast<py::array>();
            out.append(this_array->intarray_getitem(input_unwrap)->unwrap());
        }
        return out;
    }

    JaggedArray* getitem(JaggedArray* input) {
        if (input->len() == 0) {
            return this;
        }
        AnyArray* content_ = input->get_content();
        while (true) {
            NumpyArray* array_content = dynamic_cast<NumpyArray*>(content_);
            if (array_content != 0) {
                if (array_content->len() == 0) {
                    return this;
                }
                if (array_content->request().format.find("?") != std::string::npos) {
                    return fromiter(booljagged_getitem(input));
                }
                return fromiter(intjagged_getitem(input));
            }
            else {
                JaggedArray* this_content = dynamic_cast<JaggedArray*>(content_);
                if (this_content != 0) {
                    content_ = this_content->get_content();
                }
                else {
                    throw std::invalid_argument("could not determine jagged array type");
                }
            }
        }
    }

    py::object python_getitem(JaggedArray* input) {
        return getitem(input)->unwrap();
    }

    py::object tolist() {
        py::list out;
        for (ssize_t i = 0; i < len(); i++) {
            out.append(getitem(i)->tolist());
        }
        return out;
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
