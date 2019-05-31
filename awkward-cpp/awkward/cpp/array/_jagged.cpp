#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cinttypes>
namespace py = pybind11;

py::array_t<std::int64_t> offsets2parents_int64(py::array_t<std::int64_t> offsets) {
    py::buffer_info offsets_info = offsets.request();
    if (offsets_info.size <= 0) {
        throw std::invalid_argument("offsets must have at least one element");
    }
    auto offsets_ptr = (std::int64_t*)offsets_info.ptr;
    
    size_t parents_length = (size_t)offsets_ptr[offsets_info.size - 1];
    auto parents = py::array_t<std::int64_t>(parents_length);
    py::buffer_info parents_info = parents.request();
    
    auto parents_ptr = (std::int64_t*)parents_info.ptr;
    
    size_t j = 0;
    size_t k = -1;
    for (size_t i = 0; i < (size_t)offsets_info.size; i++) {
        while (j < (size_t)offsets_ptr[i]) {
            parents_ptr[j] = k;
            j += 1;
        }
        k += 1;
    }
    
    return parents;
}

py::array_t<std::int32_t> offsets2parents_int32(py::array_t<std::int32_t> offsets) {
    py::buffer_info offsets_info = offsets.request();
    if (offsets_info.size <= 0) {
        throw std::invalid_argument("offsets must have at least one element");
    }
    auto offsets_ptr = (std::int32_t*)offsets_info.ptr;
    
    size_t parents_length = offsets_ptr[offsets_info.size - 1];
    auto parents = py::array_t<std::int32_t>(parents_length);
    py::buffer_info parents_info = parents.request();
    
    auto parents_ptr = (std::int32_t*)parents_info.ptr;
    
    size_t j = 0;
    size_t k = -1;
    for (size_t i = 0; i < (size_t)offsets_info.size; i++) {
        while (j < (size_t)offsets_ptr[i]) {
            parents_ptr[j] = k;
            j += 1;
        }
        k += 1;
    }
    
    return parents;
}

py::array_t<std::int64_t> counts2offsets_int64(py::array_t<std::int64_t> counts) {
    py::buffer_info counts_info = counts.request();
    auto counts_ptr = (std::int64_t*)counts_info.ptr;
    
    size_t offsets_length = counts_info.size + 1;
    auto offsets = py::array_t<std::int64_t>(offsets_length);
    py::buffer_info offsets_info = offsets.request();
    auto offsets_ptr = (std::int64_t*)offsets_info.ptr;
    
    offsets_ptr[0] = 0;
    for (size_t i = 0; i < (size_t)counts_info.size; i++) {
        offsets_ptr[i + 1] = offsets_ptr[i] + counts_ptr[i];
    }
    return offsets;
}

py::array_t<std::int32_t> counts2offsets_int32(py::array_t<std::int32_t> counts) {
    py::buffer_info counts_info = counts.request();
    auto counts_ptr = (std::int32_t*)counts_info.ptr;
    
    size_t offsets_length = counts_info.size + 1;
    auto offsets = py::array_t<std::int32_t>(offsets_length);
    py::buffer_info offsets_info = offsets.request();
    auto offsets_ptr = (std::int32_t*)offsets_info.ptr;
    
    offsets_ptr[0] = 0;
    for (size_t i = 0; i < (size_t)counts_info.size; i++) {
        offsets_ptr[i + 1] = offsets_ptr[i] + counts_ptr[i];
    }
    return offsets;
}

py::array_t<std::int64_t> startsstops2parents_int64(py::array_t<std::int64_t> starts, py::array_t<std::int64_t> stops) {
    py::buffer_info starts_info = starts.request();
    auto starts_ptr = (std::int64_t*)starts_info.ptr;
    
    py::buffer_info stops_info = stops.request();
    auto stops_ptr = (std::int64_t*)stops_info.ptr;
    
    size_t max;
    if (stops_info.size < 1) {
        max = 0;
    }
    else {
        max = (size_t)stops_ptr[0];
        for (size_t i = 1; i < (size_t)stops_info.size; i++) {
            if ((size_t)stops_ptr[i] > max) {
                max = (size_t)stops_ptr[i];
            }
        }
    }
    auto parents = py::array_t<std::int64_t>(max);
    py::buffer_info parents_info = parents.request();
    auto parents_ptr = (std::int64_t*)parents_info.ptr;
    for (size_t i = 0; i < max; i++) {
        parents_ptr[i] = -1;
    }
    
    for (size_t i = 0; i < (size_t)starts_info.size; i++) {
        for (size_t j = (size_t)starts_ptr[i]; j < (size_t)stops_ptr[i]; j++) {
            parents_ptr[j] = i;
        }
    }
    
    return parents;
}

py::array_t<std::int32_t> startsstops2parents_int32(py::array_t<std::int32_t> starts, py::array_t<std::int32_t> stops) {
    py::buffer_info starts_info = starts.request();
    auto starts_ptr = (std::int32_t*)starts_info.ptr;
    
    py::buffer_info stops_info = stops.request();
    auto stops_ptr = (std::int32_t*)stops_info.ptr;
    
    size_t max;
    if (stops_info.size < 1) {
        max = 0;
    }
    else {
        max = (size_t)stops_ptr[0];
        for (size_t i = 1; i < (size_t)stops_info.size; i++) {
            if ((size_t)stops_ptr[i] > max) {
                max = (size_t)stops_ptr[i];
            }
        }
    }
    auto parents = py::array_t<std::int32_t>(max);
    py::buffer_info parents_info = parents.request();
    auto parents_ptr = (std::int32_t*)parents_info.ptr;
    for (size_t i = 0; i < max; i++) {
        parents_ptr[i] = -1;
    }
    
    for (size_t i = 0; i < (size_t)starts_info.size; i++) {
        for (size_t j = (size_t)starts_ptr[i]; j < (size_t)stops_ptr[i]; j++) {
            parents_ptr[j] = i;
        }
    }
    
    return parents;
}

py::tuple parents2startsstops_int64(py::array_t<std::int64_t> parents, size_t length) {
    py::buffer_info parents_info = parents.request();
    auto parents_ptr = (std::int64_t*)parents_info.ptr;
    
    auto starts = py::array_t<std::int64_t>(length);
    py::buffer_info starts_info = starts.request();
    auto starts_ptr = (std::int64_t*)starts_info.ptr;
    
    auto stops = py::array_t<std::int64_t>(length);
    py::buffer_info stops_info = stops.request();
    auto stops_ptr = (std::int64_t*)stops_info.ptr;

    for (size_t i = 0; i < length; i++) {
        starts_ptr[i] = 0;
        stops_ptr[i] = 0;
    }
    
    std::int64_t last = -1;
    for (size_t k = 0; k < (size_t)parents_info.size; k++) {
        std::int64_t thisOne = (std::int64_t)parents_ptr[k];
        if (last != thisOne) {
            if (last >= 0 && last < (std::int64_t)length) {
                stops_ptr[last] = (std::int64_t)k;
            }
            if (thisOne >= 0 && thisOne < (std::int64_t)length) {
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

py::tuple parents2startsstops_int32(py::array_t<std::int32_t> parents, size_t length) {
    py::buffer_info parents_info = parents.request();
    auto parents_ptr = (std::int32_t*)parents_info.ptr;

    auto starts = py::array_t<std::int32_t>(length);
    py::buffer_info starts_info = starts.request();
    auto starts_ptr = (std::int32_t*)starts_info.ptr;

    auto stops = py::array_t<std::int32_t>(length);
    py::buffer_info stops_info = stops.request();
    auto stops_ptr = (std::int32_t*)stops_info.ptr;

    for (size_t i = 0; i < length; i++) {
        starts_ptr[i] = 0;
        stops_ptr[i] = 0;
    }

    std::int32_t last = -1;
    for (size_t k = 0; k < (size_t)parents_info.size; k++) {
        std::int32_t thisOne = (std::int32_t)parents_ptr[k];
        if (last != thisOne) {
            if (last >= 0 && last < (std::int32_t)length) {
                stops_ptr[last] = (std::int32_t)k;
            }
            if (thisOne >= 0 && thisOne < (std::int32_t)length) {
                starts_ptr[thisOne] = (std::int32_t)k;
            }
        }
        last = thisOne;
    }

    if (last != -1) {
        stops_ptr[last] = (std::int32_t)parents_info.size;
    }

    py::list temp;
    temp.append(starts);
    temp.append(stops);
    py::tuple out(temp);
    return out;
}

py::tuple uniques2offsetsparents_int64(py::array_t<std::int64_t> uniques) {
    py::buffer_info uniques_info = uniques.request();
    auto uniques_ptr = (std::int64_t*)uniques_info.ptr;

    size_t tempLength;
    if (uniques_info.size < 1) {
        tempLength = 0;
    }
    else {
        tempLength = uniques_info.size - 1;
    }
    auto tempArray = py::array_t<bool>(tempLength);
    py::buffer_info tempArray_info = tempArray.request();
    auto tempArray_ptr = (bool*)tempArray_info.ptr;

    size_t countLength = 0;
    for (size_t i = 0; i < (size_t)uniques_info.size - 1; i++) {
        if (uniques_ptr[i] != uniques_ptr[i + 1]) {
            tempArray_ptr[i] = true;
            countLength++;
        }
        else {
            tempArray_ptr[i] = false;
        }
    }
    auto changes = py::array_t<int64_t>(countLength);
    py::buffer_info changes_info = changes.request();
    auto changes_ptr = (std::int64_t*)changes_info.ptr;
    size_t index = 0;
    for (size_t i = 0; i < (size_t)tempArray_info.size; i++) {
        if (tempArray_ptr[i]) {
            changes_ptr[index++] = (std::int64_t)(i + 1);
        }
    }

    auto offsets = py::array_t<int64_t>(changes_info.size + 2);
    py::buffer_info offsets_info = offsets.request();
    auto offsets_ptr = (std::int64_t*)offsets_info.ptr;
    offsets_ptr[0] = 0;
    offsets_ptr[offsets_info.size - 1] = (std::int64_t)uniques_info.size;
    for (size_t i = 1; i < (size_t)offsets_info.size - 1; i++) {
        offsets_ptr[i] = changes_ptr[i - 1];
    }

    auto parents = py::array_t<int64_t>(uniques_info.size);
    py::buffer_info parents_info = parents.request();
    auto parents_ptr = (std::int64_t*)parents_info.ptr;
    for (size_t i = 0; i < (size_t)parents_info.size; i++) {
        parents_ptr[i] = 0;
    }
    for (size_t i = 0; i < (size_t)changes_info.size; i++) {
        parents_ptr[(size_t)changes_ptr[i]] = 1;
    }
    for (size_t i = 1; i < (size_t)parents_info.size; i++) {
        parents_ptr[i] += parents_ptr[i - 1];
    }

    py::list temp;
    temp.append(offsets);
    temp.append(parents);
    py::tuple out(temp);
    return out;
}

py::tuple uniques2offsetsparents_int32(py::array_t<std::int32_t> uniques) {
    py::buffer_info uniques_info = uniques.request();
    auto uniques_ptr = (std::int32_t*)uniques_info.ptr;

    size_t tempLength;
    if (uniques_info.size < 1) {
        tempLength = 0;
    }
    else {
        tempLength = uniques_info.size - 1;
    }
    auto tempArray = py::array_t<bool>(tempLength);
    py::buffer_info tempArray_info = tempArray.request();
    auto tempArray_ptr = (bool*)tempArray_info.ptr;

    size_t countLength = 0;
    for (size_t i = 0; i < (size_t)uniques_info.size - 1; i++) {
        if (uniques_ptr[i] != uniques_ptr[i + 1]) {
            tempArray_ptr[i] = true;
            countLength++;
        }
        else {
            tempArray_ptr[i] = false;
        }
    }
    auto changes = py::array_t<int32_t>(countLength);
    py::buffer_info changes_info = changes.request();
    auto changes_ptr = (std::int32_t*)changes_info.ptr;
    size_t index = 0;
    for (size_t i = 0; i < (size_t)tempArray_info.size; i++) {
        if (tempArray_ptr[i]) {
            changes_ptr[index++] = (std::int32_t)(i + 1);
        }
    }

    auto offsets = py::array_t<int32_t>(changes_info.size + 2);
    py::buffer_info offsets_info = offsets.request();
    auto offsets_ptr = (std::int32_t*)offsets_info.ptr;
    offsets_ptr[0] = 0;
    offsets_ptr[offsets_info.size - 1] = (std::int32_t)uniques_info.size;
    for (size_t i = 1; i < (size_t)offsets_info.size - 1; i++) {
        offsets_ptr[i] = changes_ptr[i - 1];
    }

    auto parents = py::array_t<int32_t>(uniques_info.size);
    py::buffer_info parents_info = parents.request();
    auto parents_ptr = (std::int32_t*)parents_info.ptr;
    for (size_t i = 0; i < (size_t)parents_info.size; i++) {
        parents_ptr[i] = 0;
    }
    for (size_t i = 0; i < (size_t)changes_info.size; i++) {
        parents_ptr[(size_t)changes_ptr[i]] = 1;
    }
    for (size_t i = 1; i < (size_t)parents_info.size; i++) {
        parents_ptr[i] += parents_ptr[i - 1];
    }

    py::list temp;
    temp.append(offsets);
    temp.append(parents);
    py::tuple out(temp);
    return out;
}

PYBIND11_MODULE(_jagged, m) {
    m.def("offsets2parents_int64", &offsets2parents_int64, "");
    m.def("offsets2parents_int32", &offsets2parents_int64, "");
    m.def("counts2offsets_int64", &counts2offsets_int64, "");
    m.def("counts2offsets_int32", &counts2offsets_int32, "");
    m.def("startsstops2parents_int64", &startsstops2parents_int64, "");
    m.def("startsstops2parents_int32", &startsstops2parents_int32, "");
    m.def("parents2startsstops_int64", &parents2startsstops_int64, "");
    m.def("parents2startsstops_int32", &parents2startsstops_int32, "");
    m.def("uniques2offsetsparents_int64", &uniques2offsetsparents_int64, "");
    m.def("uniques2offsetsparents_int32", &uniques2offsetsparents_int32, "");
}
