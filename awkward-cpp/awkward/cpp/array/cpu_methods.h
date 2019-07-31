#ifdef __cplusplus           // specifies that this is C, not C++
extern "C" {
#endif

#ifndef CPU_METHODS_H        // include guard
#define CPU_METHODS_H

#include <stdint.h>

struct c_array {             // this is the C structure of pybind11::buffer_info
    void          *ptr;
    ssize_t        itemsize;
    ssize_t        size;
    const char    *format;
    ssize_t        ndim;
    const ssize_t *shape;
    const ssize_t *strides;
};

int byteswap_16bit(int16_t *val) {
    *val = *val << 8 | *val >> 8;
    return 1;
}

int byteswap_32bit(int32_t *val) {
    *val = ((*val << 8) & 0xFF00FF00) | ((*val >> 8) & 0xFF00FF);
    *val = (*val << 16) | (*val >> 16);
    return 1;
}

int byteswap_64bit(int64_t *val) {
    *val = ((*val << 8) & 0xFF00FF00FF00FF00ULL) | ((*val >> 8) & 0x00FF00FF00FF00FFULL);
    *val = ((*val << 16) & 0xFFFF0000FFFF0000ULL) | ((*val >> 16) & 0x0000FFFF0000FFFFULL);
    *val = (*val << 32) | (*val >> 32);
    return 1;
}

int isNative_CPU(struct c_array *input) {
    // returns 1 if input is native, or 0 if non-isNative
    union {
        uint32_t i;
        char c[4];
    } bint = { 0x01020304 };
    return ((bint.c[0] == 1 && input->format[0] != '<')
        || (bint.c[0] != 1 && input->format[0] != '>'));
}

int makeNative_16bit(struct c_array *input, ssize_t dim, ssize_t index) {
    // to be initially called with makeNative_16bit(input, 0, 0)
    if (dim > input->ndim - 1)
        return 0;
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            byteswap_16bit(&((int16_t*)input->ptr)[index + i * N]);
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        makeNative_16bit(input, dim + 1, index + i * N);
    return 1;
}

int makeNative_32bit(struct c_array *input, ssize_t dim, ssize_t index) {
    // to be initially called with makeNative_32bit(input, 0, 0)
    if (dim > input->ndim - 1)
        return 0;
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            byteswap_32bit(&((int32_t*)input->ptr)[index + i * N]);
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        makeNative_32bit(input, dim + 1, index + i * N);
    return 1;
}

int makeNative_64bit(struct c_array *input, ssize_t dim, ssize_t index) {
    // to be initially called with makeNative_64bit(input, 0, 0)
    if (dim > input->ndim - 1)
        return 0;
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            byteswap_64bit(&((int64_t*)input->ptr)[index + i * N]);
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        makeNative_64bit(input, dim + 1, index + i * N);
    return 1;
}

int makeNative_CPU(struct c_array *input) {
    /* PURPOSE:
        - checks if input array is a native array
        - if it is non-native, it makes it native
    PREREQUISITES:
        - can be an array of any dimensionality
    */
    if (isNative_CPU(input))
        return 1;
    if (input->itemsize == 1)
        return 1;
    if (input->itemsize == 2)
        return makeNative_16bit(input, 0, 0);
    if (input->itemsize == 4)
        return makeNative_32bit(input, 0, 0);
    if (input->itemsize == 8)
        return makeNative_64bit(input, 0, 0);
    return 0;
}

int checkInt_CPU(struct c_array *input) {
    // returns 1 if it's an int array
    if (input->size == 0) {
        return 1;
    }
    char intList[] = "qQlLhHbB";
    ssize_t k = 0;
    if (input->format[0] == '<' || input->format[0] == '>' ||
        input->format[0] == '=')
        k = 1;
    for (ssize_t i = 0; i < 8; i++)
        if (intList[i] == input->format[k])
            return 1;
    return 0;
}

int getMax_8bit(struct c_array *input, ssize_t dim, ssize_t index, int8_t *max) {
    // to be initially called with getMax_8bit(input, 0, 0, *0)
    if (dim > input->ndim - 1)
        return 0;
    if (dim == 0) {
        if (input->shape[0] == 0) {
            *max = 0;
            return 1;
        }
        *max = ((int8_t*)input->ptr)[0];
    }
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int8_t*)input->ptr)[index + i * N] > *max)
                *max = ((int8_t*)input->ptr)[index + i * N];
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        getMax_8bit(input, dim + 1, index + i * N, max);
    return 1;
}

int getMax_16bit(struct c_array *input, ssize_t dim, ssize_t index, int16_t *max) {
    // to be initially called with getMax_16bit(input, 0, 0, *0)
    if (dim > input->ndim - 1)
        return 0;
    if (dim == 0) {
        if (input->shape[0] == 0) {
            *max = 0;
            return 1;
        }
        *max = ((int16_t*)input->ptr)[0];
    }
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int16_t*)input->ptr)[index + i * N] > *max)
                *max = ((int16_t*)input->ptr)[index + i * N];
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        getMax_16bit(input, dim + 1, index + i * N, max);
    return 1;
}

int getMax_32bit(struct c_array *input, ssize_t dim, ssize_t index, int32_t *max) {
    // to be initially called with getMax_32bit(input, 0, 0, *0)
    if (dim > input->ndim - 1)
        return 0;
    if (dim == 0) {
        if (input->shape[0] == 0) {
            *max = 0;
            return 1;
        }
        *max = ((int32_t*)input->ptr)[0];
    }
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int32_t*)input->ptr)[index + i * N] > *max)
                *max = ((int32_t*)input->ptr)[index + i * N];
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        getMax_32bit(input, dim + 1, index + i * N, max);
    return 1;
}

int getMax_64bit(struct c_array *input, ssize_t dim, ssize_t index, int64_t *max) {
    // to be initially called with getMax_64bit(input, 0, 0, *0)
    if (dim > input->ndim - 1)
        return 0;
    if (dim == 0) {
        if (input->shape[0] == 0) {
            *max = 0;
            return 1;
        }
        *max = ((int64_t*)input->ptr)[0];
    }
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int64_t*)input->ptr)[index + i * N] > *max)
                *max = ((int64_t*)input->ptr)[index + i * N];
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        getMax_64bit(input, dim + 1, index + i * N, max);
    return 1;
}

int offsets2parents_8bit(struct c_array *offsets, struct c_array *parents) {
    ssize_t N = offsets->strides[0] / offsets->itemsize, j = 0, k = -1;
    for (ssize_t i = 0; i < offsets->size; i++) {
        while (j < (ssize_t)((int8_t*)offsets->ptr)[i * N])
            ((int8_t*)parents->ptr)[j++] = (int8_t)k;
        k++;
    }
    return 1;
}

int offsets2parents_16bit(struct c_array *offsets, struct c_array *parents) {
    ssize_t N = offsets->strides[0] / offsets->itemsize, j = 0, k = -1;
    for (ssize_t i = 0; i < offsets->size; i++) {
        while (j < (ssize_t)((int16_t*)offsets->ptr)[i * N])
            ((int16_t*)parents->ptr)[j++] = (int16_t)k;
        k++;
    }
    return 1;
}

int offsets2parents_32bit(struct c_array *offsets, struct c_array *parents) {
    ssize_t N = offsets->strides[0] / offsets->itemsize, j = 0, k = -1;
    for (ssize_t i = 0; i < offsets->size; i++) {
        while (j < (ssize_t)((int32_t*)offsets->ptr)[i * N])
            ((int32_t*)parents->ptr)[j++] = (int32_t)k;
        k++;
    }
    return 1;
}

int offsets2parents_64bit(struct c_array *offsets, struct c_array *parents) {
    ssize_t N = offsets->strides[0] / offsets->itemsize, j = 0, k = -1;
    for (ssize_t i = 0; i < offsets->size; i++) {
        while (j < (ssize_t)((int64_t*)offsets->ptr)[i * N])
            ((int64_t*)parents->ptr)[j++] = (int64_t)k;
        k++;
    }
    return 1;
}

int offsets2parents_CPU(struct c_array *offsets, struct c_array *parents) {
    /* PURPOSE:
        - converts offsets to parents
    PREREQUISITES:
        - offsets is a non-negative, 1d int array of length > 0
        - parents is a NEW array of size offsets[-1]
        - offsets and parents are of the same type
    */
    if (offsets->itemsize == 1)
        return offsets2parents_8bit(offsets, parents);
    if (offsets->itemsize == 2)
        return offsets2parents_16bit(offsets, parents);
    if (offsets->itemsize == 4)
        return offsets2parents_32bit(offsets, parents);
    if (offsets->itemsize == 8)
        return offsets2parents_64bit(offsets, parents);
    return 0;
}

int counts2offsets_8bit(struct c_array *counts, struct c_array *offsets) {
    ssize_t N = counts->strides[0] / counts->itemsize;
    ((int8_t*)offsets->ptr)[0] = 0;
    for (ssize_t i = 0; i < counts->size; i++)
        ((int8_t*)offsets->ptr)[i + 1] = ((int8_t*)offsets->ptr)[i] + ((int8_t*)counts->ptr)[i * N];
    return 1;
}

int counts2offsets_16bit(struct c_array *counts, struct c_array *offsets) {
    ssize_t N = counts->strides[0] / counts->itemsize;
    ((int16_t*)offsets->ptr)[0] = 0;
    for (ssize_t i = 0; i < counts->size; i++)
        ((int16_t*)offsets->ptr)[i + 1] = ((int16_t*)offsets->ptr)[i] + ((int16_t*)counts->ptr)[i * N];
    return 1;
}

int counts2offsets_32bit(struct c_array *counts, struct c_array *offsets) {
    ssize_t N = counts->strides[0] / counts->itemsize;
    ((int32_t*)offsets->ptr)[0] = 0;
    for (ssize_t i = 0; i < counts->size; i++)
        ((int32_t*)offsets->ptr)[i + 1] = ((int32_t*)offsets->ptr)[i] + ((int32_t*)counts->ptr)[i * N];
    return 1;
}

int counts2offsets_64bit(struct c_array *counts, struct c_array *offsets) {
    ssize_t N = counts->strides[0] / counts->itemsize;
    ((int64_t*)offsets->ptr)[0] = 0;
    for (ssize_t i = 0; i < counts->size; i++)
        ((int64_t*)offsets->ptr)[i + 1] = ((int64_t*)offsets->ptr)[i] + ((int64_t*)counts->ptr)[i * N];
    return 1;
}

int counts2offsets_CPU(struct c_array *counts, struct c_array *offsets) {
    /* PURPOSE:
        - converts counts to offsets
    PREREQUISITES:
        - counts is a non-negative, 1d int array
        - offsets is a NEW array of size offsets->size + 1
        - counts and offsets are of the same type
        - note: for >1d counts, flatten it first
    */
    if (counts->itemsize == 1)
        return counts2offsets_8bit(counts, offsets);
    if (counts->itemsize == 2)
        return counts2offsets_16bit(counts, offsets);
    if (counts->itemsize == 4)
        return counts2offsets_32bit(counts, offsets);
    if (counts->itemsize == 8)
        return counts2offsets_64bit(counts, offsets);
    return 0;
}

int startsstops2parents_8bit(struct c_array *starts, struct c_array *stops, struct c_array *parents) {
    for (ssize_t i = 0; i < parents->size; i++)
        ((int8_t*)parents->ptr)[i] = -1;
    ssize_t N_sta = starts->strides[0] / starts->itemsize, N_sto = stops->strides[0] / stops->itemsize;
    for (ssize_t i = 0; i < starts->size; i++)
        for (ssize_t j = (ssize_t)((int8_t*)starts->ptr)[i * N_sta]; j < (ssize_t)((int8_t*)stops->ptr)[i * N_sto]; j++)
            ((int8_t*)parents->ptr)[j] = (int8_t)i;
    return 1;
}

int startsstops2parents_16bit(struct c_array *starts, struct c_array *stops, struct c_array *parents) {
    for (ssize_t i = 0; i < parents->size; i++)
        ((int16_t*)parents->ptr)[i] = -1;
    ssize_t N_sta = starts->strides[0] / starts->itemsize, N_sto = stops->strides[0] / stops->itemsize;
    for (ssize_t i = 0; i < starts->size; i++)
        for (ssize_t j = (ssize_t)((int16_t*)starts->ptr)[i * N_sta]; j < (ssize_t)((int16_t*)stops->ptr)[i * N_sto]; j++)
            ((int16_t*)parents->ptr)[j] = (int16_t)i;
    return 1;
}

int startsstops2parents_32bit(struct c_array *starts, struct c_array *stops, struct c_array *parents) {
    for (ssize_t i = 0; i < parents->size; i++)
        ((int32_t*)parents->ptr)[i] = -1;
    ssize_t N_sta = starts->strides[0] / starts->itemsize, N_sto = stops->strides[0] / stops->itemsize;
    for (ssize_t i = 0; i < starts->size; i++)
        for (ssize_t j = (ssize_t)((int32_t*)starts->ptr)[i * N_sta]; j < (ssize_t)((int32_t*)stops->ptr)[i * N_sto]; j++)
            ((int32_t*)parents->ptr)[j] = (int32_t)i;
    return 1;
}

int startsstops2parents_64bit(struct c_array *starts, struct c_array *stops, struct c_array *parents) {
    for (ssize_t i = 0; i < parents->size; i++)
        ((int64_t*)parents->ptr)[i] = -1;
    ssize_t N_sta = starts->strides[0] / starts->itemsize, N_sto = stops->strides[0] / stops->itemsize;
    for (ssize_t i = 0; i < starts->size; i++)
        for (ssize_t j = (ssize_t)((int64_t*)starts->ptr)[i * N_sta]; j < (ssize_t)((int64_t*)stops->ptr)[i * N_sto]; j++)
            ((int64_t*)parents->ptr)[j] = (int64_t)i;
    return 1;
}

int startsstops2parents_CPU(struct c_array *starts, struct c_array *stops, struct c_array *parents) {
    /* PURPOSE:
        - converts starts and stops to parents
    PREREQUISITES:
        - starts and stops are non-negative 1d int arrays
        - starts must be less than or equal to stops
        - starts must be of equal or shorter length than stops
        - parents is a NEW array of length stops->maximum
        - starts, stops, and parents are of the same type
        - note: for >1d starts/stops, flatten them first
    */
    if (starts->itemsize == 1)
        return startsstops2parents_8bit(starts, stops, parents);
    if (starts->itemsize == 2)
        return startsstops2parents_16bit(starts, stops, parents);
    if (starts->itemsize == 4)
        return startsstops2parents_32bit(starts, stops, parents);
    if (starts->itemsize == 8)
        return startsstops2parents_64bit(starts, stops, parents);
    return 0;
}

int parents2startsstops_8bit(struct c_array *parents, struct c_array *starts, struct c_array *stops) {
    for (ssize_t i = 0; i < starts->size; i++) {
        ((int8_t*)starts->ptr)[i] = 0;
        ((int8_t*)stops->ptr)[i] = 0;
    }
    ssize_t N = parents->strides[0] / parents->itemsize;
    int8_t last = -1;
    for (ssize_t i = 0; i < parents->size; i++) {
        int8_t thisOne = ((int8_t*)parents->ptr)[i * N];
        if (last != thisOne) {
            if (last >= 0 && last < starts->size)
                ((int8_t*)stops->ptr)[(ssize_t)last] = (int8_t)i;
            if (thisOne >= 0 && thisOne < starts->size)
                ((int8_t*)starts->ptr)[(ssize_t)thisOne] = (int8_t)i;
        }
        last = thisOne;
    }
    if (last != -1)
        ((int8_t*)stops->ptr)[(ssize_t)last] = (int8_t)parents->size;
    return 1;
}

int parents2startsstops_16bit(struct c_array *parents, struct c_array *starts, struct c_array *stops) {
    for (ssize_t i = 0; i < starts->size; i++) {
        ((int16_t*)starts->ptr)[i] = 0;
        ((int16_t*)stops->ptr)[i] = 0;
    }
    ssize_t N = parents->strides[0] / parents->itemsize;
    int16_t last = -1;
    for (ssize_t i = 0; i < parents->size; i++) {
        int16_t thisOne = ((int16_t*)parents->ptr)[i * N];
        if (last != thisOne) {
            if (last >= 0 && last < starts->size)
                ((int16_t*)stops->ptr)[(ssize_t)last] = (int16_t)i;
            if (thisOne >= 0 && thisOne < starts->size)
                ((int16_t*)starts->ptr)[(ssize_t)thisOne] = (int16_t)i;
        }
        last = thisOne;
    }
    if (last != -1)
        ((int16_t*)stops->ptr)[(ssize_t)last] = (int16_t)parents->size;
    return 1;
}

int parents2startsstops_32bit(struct c_array *parents, struct c_array *starts, struct c_array *stops) {
    for (ssize_t i = 0; i < starts->size; i++) {
        ((int32_t*)starts->ptr)[i] = 0;
        ((int32_t*)stops->ptr)[i] = 0;
    }
    ssize_t N = parents->strides[0] / parents->itemsize;
    int32_t last = -1;
    for (ssize_t i = 0; i < parents->size; i++) {
        int32_t thisOne = ((int32_t*)parents->ptr)[i * N];
        if (last != thisOne) {
            if (last >= 0 && last < starts->size)
                ((int32_t*)stops->ptr)[(ssize_t)last] = (int32_t)i;
            if (thisOne >= 0 && thisOne < starts->size)
                ((int32_t*)starts->ptr)[(ssize_t)thisOne] = (int32_t)i;
        }
        last = thisOne;
    }
    if (last != -1)
        ((int32_t*)stops->ptr)[(ssize_t)last] = (int32_t)parents->size;
    return 1;
}

int parents2startsstops_64bit(struct c_array *parents, struct c_array *starts, struct c_array *stops) {
    for (ssize_t i = 0; i < starts->size; i++) {
        ((int64_t*)starts->ptr)[i] = 0;
        ((int64_t*)stops->ptr)[i] = 0;
    }
    ssize_t N = parents->strides[0] / parents->itemsize;
    int64_t last = -1;
    for (ssize_t i = 0; i < parents->size; i++) {
        int64_t thisOne = ((int64_t*)parents->ptr)[i * N];
        if (last != thisOne) {
            if (last >= 0 && last < starts->size)
                ((int64_t*)stops->ptr)[(ssize_t)last] = (int64_t)i;
            if (thisOne >= 0 && thisOne < starts->size)
                ((int64_t*)starts->ptr)[(ssize_t)thisOne] = (int64_t)i;
        }
        last = thisOne;
    }
    if (last != -1)
        ((int64_t*)stops->ptr)[(ssize_t)last] = (int64_t)parents->size;
    return 1;
}

int parents2startsstops_CPU(struct c_array *parents, struct c_array *starts, struct c_array *stops) {
    /* PURPOSE:
        - converts parents to starts and stops
    PREREQUISITES:
        - parents is a 1d int array
        - starts and stops are NEW arrays of length <= parents->max + 1
        - parents, starts, and stops are of the same type
    */
    if (parents->itemsize == 1)
        return parents2startsstops_8bit(parents, starts, stops);
    if (parents->itemsize == 2)
        return parents2startsstops_16bit(parents, starts, stops);
    if (parents->itemsize == 4)
        return parents2startsstops_32bit(parents, starts, stops);
    if (parents->itemsize == 8)
        return parents2startsstops_64bit(parents, starts, stops);
    return 0;
}

int uniques2offsetsparents_generateTemparray_8bit(struct c_array *uniques, struct c_array *tempArray, ssize_t *countLength) {
    *countLength = 0;
    ssize_t N = uniques->strides[0] / uniques->itemsize;
    for (ssize_t i = 0; i < uniques->size - 1; i++) {
        if (((int8_t*)uniques->ptr)[i * N] != ((int8_t*)uniques->ptr)[(i + 1) * N]) {
            ((int8_t*)tempArray->ptr)[i] = 1;
            (*countLength)++;
        }
        else
            ((int8_t*)tempArray->ptr)[i] = 0;
    }
    return 1;
}

int uniques2offsetsparents_generateTemparray_16bit(struct c_array *uniques, struct c_array *tempArray, ssize_t *countLength) {
    *countLength = 0;
    ssize_t N = uniques->strides[0] / uniques->itemsize;
    for (ssize_t i = 0; i < uniques->size - 1; i++) {
        if (((int16_t*)uniques->ptr)[i * N] != ((int16_t*)uniques->ptr)[(i + 1) * N]) {
            ((int8_t*)tempArray->ptr)[i] = 1;
            (*countLength)++;
        }
        else
            ((int8_t*)tempArray->ptr)[i] = 0;
    }
    return 1;
}

int uniques2offsetsparents_generateTemparray_32bit(struct c_array *uniques, struct c_array *tempArray, ssize_t *countLength) {
    *countLength = 0;
    ssize_t N = uniques->strides[0] / uniques->itemsize;
    for (ssize_t i = 0; i < uniques->size - 1; i++) {
        if (((int32_t*)uniques->ptr)[i * N] != ((int32_t*)uniques->ptr)[(i + 1) * N]) {
            ((int8_t*)tempArray->ptr)[i] = 1;
            (*countLength)++;
        }
        else
            ((int8_t*)tempArray->ptr)[i] = 0;
    }
    return 1;
}

int uniques2offsetsparents_generateTemparray_64bit(struct c_array *uniques, struct c_array *tempArray, ssize_t *countLength) {
    *countLength = 0;
    ssize_t N = uniques->strides[0] / uniques->itemsize;
    for (ssize_t i = 0; i < uniques->size - 1; i++) {
        if (((int64_t*)uniques->ptr)[i * N] != ((int64_t*)uniques->ptr)[(i + 1) * N]) {
            ((int8_t*)tempArray->ptr)[i] = 1;
            (*countLength)++;
        }
        else
            ((int8_t*)tempArray->ptr)[i] = 0;
    }
    return 1;
}

int uniques2offsetsparents_generateTemparray_CPU(struct c_array *uniques, struct c_array *tempArray, ssize_t *countLength) {
    /* PURPOSE:
        - fills temparray and countLength as part of uniques2offsetsparents
    PREREQUISITES:
        - uniques is a 1d int array
        - tempArray is a NEW int8_t array of length: [uniques->size - 1] (or 0 at minimum)
    */
    if (uniques->itemsize == 1)
        return uniques2offsetsparents_generateTemparray_8bit(uniques, tempArray, countLength);
    if (uniques->itemsize == 2)
        return uniques2offsetsparents_generateTemparray_16bit(uniques, tempArray, countLength);
    if (uniques->itemsize == 4)
        return uniques2offsetsparents_generateTemparray_32bit(uniques, tempArray, countLength);
    if (uniques->itemsize == 8)
        return uniques2offsetsparents_generateTemparray_64bit(uniques, tempArray, countLength);
    return 0;
}

int uniques2offsetsparents_8bit(ssize_t countLength, struct c_array *tempArray, struct c_array *offsets, struct c_array *parents) {
    ssize_t *changes;
    changes = (ssize_t*) calloc(countLength, sizeof(ssize_t));
    ssize_t index = 0;
    for (ssize_t i = 0; i < tempArray->size; i++)
        if (((int8_t*)tempArray->ptr)[i])
            changes[index++] = i + 1;
    ((int8_t*)offsets->ptr)[0] = 0;
    ((int8_t*)offsets->ptr)[offsets->size - 1] = (int8_t)parents->size;
    for (ssize_t i = 1; i < offsets->size - 1; i++)
        ((int8_t*)offsets->ptr)[i] = (int8_t)changes[i - 1];
    for (ssize_t i = 0; i < parents->size; i++)
        ((int8_t*)parents->ptr)[i] = 0;
    for (ssize_t i = 0; i < countLength; i++)
        ((int8_t*)parents->ptr)[changes[i]] = 1;
    for (ssize_t i = 1; i < parents->size; i++)
        ((int8_t*)parents->ptr)[i] += ((int8_t*)parents->ptr)[i - 1];
    free(changes);
    return 1;
}

int uniques2offsetsparents_16bit(ssize_t countLength, struct c_array *tempArray, struct c_array *offsets, struct c_array *parents) {
    ssize_t *changes;
    changes = (ssize_t*) calloc(countLength, sizeof(ssize_t));
    ssize_t index = 0;
    for (ssize_t i = 0; i < tempArray->size; i++)
        if (((int8_t*)tempArray->ptr)[i])
            changes[index++] = i + 1;
    ((int16_t*)offsets->ptr)[0] = 0;
    ((int16_t*)offsets->ptr)[offsets->size - 1] = (int16_t)parents->size;
    for (ssize_t i = 1; i < offsets->size - 1; i++)
        ((int16_t*)offsets->ptr)[i] = (int16_t)changes[i - 1];
    for (ssize_t i = 0; i < parents->size; i++)
        ((int16_t*)parents->ptr)[i] = 0;
    for (ssize_t i = 0; i < countLength; i++)
        ((int16_t*)parents->ptr)[changes[i]] = 1;
    for (ssize_t i = 1; i < parents->size; i++)
        ((int16_t*)parents->ptr)[i] += ((int16_t*)parents->ptr)[i - 1];
    free(changes);
    return 1;
}

int uniques2offsetsparents_32bit(ssize_t countLength, struct c_array *tempArray, struct c_array *offsets, struct c_array *parents) {
    ssize_t *changes;
    changes = (ssize_t*) calloc(countLength, sizeof(ssize_t));
    ssize_t index = 0;
    for (ssize_t i = 0; i < tempArray->size; i++)
        if (((int8_t*)tempArray->ptr)[i])
            changes[index++] = i + 1;
    ((int32_t*)offsets->ptr)[0] = 0;
    ((int32_t*)offsets->ptr)[offsets->size - 1] = (int32_t)parents->size;
    for (ssize_t i = 1; i < offsets->size - 1; i++)
        ((int32_t*)offsets->ptr)[i] = (int32_t)changes[i - 1];
    for (ssize_t i = 0; i < parents->size; i++)
        ((int32_t*)parents->ptr)[i] = 0;
    for (ssize_t i = 0; i < countLength; i++)
        ((int32_t*)parents->ptr)[changes[i]] = 1;
    for (ssize_t i = 1; i < parents->size; i++)
        ((int32_t*)parents->ptr)[i] += ((int32_t*)parents->ptr)[i - 1];
    free(changes);
    return 1;
}

int uniques2offsetsparents_64bit(ssize_t countLength, struct c_array *tempArray, struct c_array *offsets, struct c_array *parents) {
    ssize_t *changes;
    changes = (ssize_t*) calloc(countLength, sizeof(ssize_t));
    ssize_t index = 0;
    for (ssize_t i = 0; i < tempArray->size; i++)
        if (((int8_t*)tempArray->ptr)[i])
            changes[index++] = i + 1;
    ((int64_t*)offsets->ptr)[0] = 0;
    ((int64_t*)offsets->ptr)[offsets->size - 1] = (int64_t)parents->size;
    for (ssize_t i = 1; i < offsets->size - 1; i++)
        ((int64_t*)offsets->ptr)[i] = (int64_t)changes[i - 1];
    for (ssize_t i = 0; i < parents->size; i++)
        ((int64_t*)parents->ptr)[i] = 0;
    for (ssize_t i = 0; i < countLength; i++)
        ((int64_t*)parents->ptr)[changes[i]] = 1;
    for (ssize_t i = 1; i < parents->size; i++)
        ((int64_t*)parents->ptr)[i] += ((int64_t*)parents->ptr)[i - 1];
    free(changes);
    return 1;
}

int uniques2offsetsparents_CPU(ssize_t countLength, struct c_array *tempArray, struct c_array *offsets, struct c_array *parents) {
    /* PURPOSE:
        - converts uniques to offsets and parents
    PREREQUISITES:
        - countLength and tempArray are from uniques2offsetsparents_generateTemparray_CPU
        - offsets is a NEW int array of length: [countLength + 2]
        - parents is a NEW int array of length: uniques->size
        - offsets and parents are of the same type
    */
    if (offsets->itemsize == 1)
        return uniques2offsetsparents_8bit(countLength, tempArray, offsets, parents);
    if (offsets->itemsize == 2)
        return uniques2offsetsparents_16bit(countLength, tempArray, offsets, parents);
    if (offsets->itemsize == 4)
        return uniques2offsetsparents_32bit(countLength, tempArray, offsets, parents);
    if (offsets->itemsize == 8)
        return uniques2offsetsparents_64bit(countLength, tempArray, offsets, parents);
    return 0;
}

int checkNonNegative_8bit(struct c_array *input, ssize_t dim, ssize_t index) {
    // to be initially called with checkNonNegative_8bit(input, 0, 0)
    if (dim > input->ndim - 1)
        return 0;
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int8_t*)input->ptr)[index + i * N] < 0)
                return 0;
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        if (!checkNonNegative_8bit(input, dim + 1, index + i * N))
            return 0;
    return 1;
}

int checkNonNegative_16bit(struct c_array *input, ssize_t dim, ssize_t index) {
    // to be initially called with checkNonNegative_16bit(input, 0, 0)
    if (dim > input->ndim - 1)
        return 0;
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int16_t*)input->ptr)[index + i * N] < 0)
                return 0;
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        if (!checkNonNegative_16bit(input, dim + 1, index + i * N))
            return 0;
    return 1;
}

int checkNonNegative_32bit(struct c_array *input, ssize_t dim, ssize_t index) {
    // to be initially called with checkNonNegative_32bit(input, 0, 0)
    if (dim > input->ndim - 1)
        return 0;
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int32_t*)input->ptr)[index + i * N] < 0)
                return 0;
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        if (!checkNonNegative_32bit(input, dim + 1, index + i * N))
            return 0;
    return 1;
}

int checkNonNegative_64bit(struct c_array *input, ssize_t dim, ssize_t index) {
    // to be initially called with checkNonNegative_64bit(input, 0, 0)
    if (dim > input->ndim - 1)
        return 0;
    ssize_t N = input->strides[dim] / input->itemsize;
    if (dim == input->ndim - 1) {
        for (ssize_t i = 0; i < input->shape[dim]; i++)
            if (((int64_t*)input->ptr)[index + i * N] < 0)
                return 0;
        return 1;
    }
    for (ssize_t i = 0; i < input->shape[dim]; i++)
        if (!checkNonNegative_64bit(input, dim + 1, index + i * N))
            return 0;
    return 1;
}

int checkNonNegative_CPU(struct c_array *input) {
    /* PURPOSE:
        - returns 1 if all non-negative, or 0 if there is a negative or an error
    PREREQUISITES:
        - input is an array, assumedly a signed int array
    */
    if (input->itemsize == 1)
        return checkNonNegative_8bit(input, 0, 0);
    if (input->itemsize == 2)
        return checkNonNegative_16bit(input, 0, 0);
    if (input->itemsize == 4)
        return checkNonNegative_32bit(input, 0, 0);
    if (input->itemsize == 8)
        return checkNonNegative_64bit(input, 0, 0);
    return 0;
}

int compare_8bit(struct c_array *a, struct c_array *b, const char *comparison, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with compare_8bit(a, b, comparison, 0, 0, 0)
    if (dim > a->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == a->ndim - 1) {
        if (comparison[0] == '>' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int8_t*)a->ptr)[index_a + i * N_a] <= ((int8_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int8_t*)a->ptr)[index_a + i * N_a] >= ((int8_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int8_t*)a->ptr)[index_a + i * N_a] != ((int8_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '>' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int8_t*)a->ptr)[index_a + i * N_a] < ((int8_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int8_t*)a->ptr)[index_a + i * N_a] > ((int8_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '!') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int8_t*)a->ptr)[index_a + i * N_a] == ((int8_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else
            return 0;
    }
    for (ssize_t i = 0; i < a->shape[dim]; i++)
        if (!compare_8bit(a, b, comparison, dim + 1, index_a + i * N_a, index_b + i * N_b))
            return 0;
    return 1;
}

int compare_16bit(struct c_array *a, struct c_array *b, const char *comparison, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with compare_16bit(a, b, comparison, 0, 0, 0)
    if (dim > a->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == a->ndim - 1) {
        if (comparison[0] == '>' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int16_t*)a->ptr)[index_a + i * N_a] <= ((int16_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int16_t*)a->ptr)[index_a + i * N_a] >= ((int16_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int16_t*)a->ptr)[index_a + i * N_a] != ((int16_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '>' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int16_t*)a->ptr)[index_a + i * N_a] < ((int16_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int16_t*)a->ptr)[index_a + i * N_a] > ((int16_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '!') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int16_t*)a->ptr)[index_a + i * N_a] == ((int16_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else
            return 0;
    }
    for (ssize_t i = 0; i < a->shape[dim]; i++)
        if (!compare_16bit(a, b, comparison, dim + 1, index_a + i * N_a, index_b + i * N_b))
            return 0;
    return 1;
}

int compare_32bit(struct c_array *a, struct c_array *b, const char *comparison, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with compare_32bit(a, b, comparison, 0, 0, 0)
    if (dim > a->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == a->ndim - 1) {
        if (comparison[0] == '>' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int32_t*)a->ptr)[index_a + i * N_a] <= ((int32_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int32_t*)a->ptr)[index_a + i * N_a] >= ((int32_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int32_t*)a->ptr)[index_a + i * N_a] != ((int32_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '>' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int32_t*)a->ptr)[index_a + i * N_a] < ((int32_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int32_t*)a->ptr)[index_a + i * N_a] > ((int32_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '!') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int32_t*)a->ptr)[index_a + i * N_a] == ((int32_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else
            return 0;
    }
    for (ssize_t i = 0; i < a->shape[dim]; i++)
        if (!compare_32bit(a, b, comparison, dim + 1, index_a + i * N_a, index_b + i * N_b))
            return 0;
    return 1;
}

int compare_64bit(struct c_array *a, struct c_array *b, const char *comparison, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with compare_64bit(a, b, comparison, 0, 0, 0)
    if (dim > a->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == a->ndim - 1) {
        if (comparison[0] == '>' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int64_t*)a->ptr)[index_a + i * N_a] <= ((int64_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] != '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int64_t*)a->ptr)[index_a + i * N_a] >= ((int64_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int64_t*)a->ptr)[index_a + i * N_a] != ((int64_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '>' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int64_t*)a->ptr)[index_a + i * N_a] < ((int64_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '<' && comparison[1] == '=') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int64_t*)a->ptr)[index_a + i * N_a] > ((int64_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else if (comparison[0] == '!') {
            for (ssize_t i = 0; i < a->shape[dim]; i++)
                if (((int64_t*)a->ptr)[index_a + i * N_a] == ((int64_t*)b->ptr)[index_b + i * N_b])
                    return 0;
            return 1;
        }
        else
            return 0;
    }
    for (ssize_t i = 0; i < a->shape[dim]; i++)
        if (!compare_64bit(a, b, comparison, dim + 1, index_a + i * N_a, index_b + i * N_b))
            return 0;
    return 1;
}

int compare_CPU(struct c_array *a, struct c_array *b, const char *comparison) {
    /* PURPOSE:
        - return 1 if a[i] [compare] b[i] for all i in a->size
        - return 0 if that isn't the case, or if there's an error
    PREREQUISITES
        - a and b must be arrays of the same type, dimensionality, and shape
        - comparison must be '>', '<', '==', '>=', '<=', or '!='
        - if comparison is not in that list, an error will occur (return 0)
        - the comparison string must be zero-terminated
    */
    if (a->itemsize == 1)
        return compare_8bit(a, b, comparison, 0, 0, 0);
    if (a->itemsize == 2)
        return compare_16bit(a, b, comparison, 0, 0, 0);
    if (a->itemsize == 4)
        return compare_32bit(a, b, comparison, 0, 0, 0);
    if (a->itemsize == 8)
        return compare_64bit(a, b, comparison, 0, 0, 0);
    return 0;
}

int deepcopy_8bit(struct c_array *a, struct c_array *b, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with deepcopy_8bit(a, b, 0, 0, 0)
    if (dim > b->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == b->ndim - 1) {
        for (ssize_t i = 0; i < b->shape[dim]; i++)
            ((int8_t*)a->ptr)[index_a + i * N_a] = ((int8_t*)b->ptr)[index_b + i * N_b];
        return 1;
    }
    for (ssize_t i = 0; i < b->shape[dim]; i++)
        deepcopy_8bit(a, b, dim + 1, index_a + i * N_a, index_b + i * N_b);
    return 1;
}

int deepcopy_16bit(struct c_array *a, struct c_array *b, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with deepcopy_16bit(a, b, 0, 0, 0)
    if (dim > b->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == b->ndim - 1) {
        for (ssize_t i = 0; i < b->shape[dim]; i++)
            ((int16_t*)a->ptr)[index_a + i * N_a] = ((int16_t*)b->ptr)[index_b + i * N_b];
        return 1;
    }
    for (ssize_t i = 0; i < b->shape[dim]; i++)
        deepcopy_16bit(a, b, dim + 1, index_a + i * N_a, index_b + i * N_b);
    return 1;
}

int deepcopy_32bit(struct c_array *a, struct c_array *b, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with deepcopy_32bit(a, b, 0, 0, 0)
    if (dim > b->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == b->ndim - 1) {
        for (ssize_t i = 0; i < b->shape[dim]; i++)
            ((int32_t*)a->ptr)[index_a + i * N_a] = ((int32_t*)b->ptr)[index_b + i * N_b];
        return 1;
    }
    for (ssize_t i = 0; i < b->shape[dim]; i++)
        deepcopy_32bit(a, b, dim + 1, index_a + i * N_a, index_b + i * N_b);
    return 1;
}

int deepcopy_64bit(struct c_array *a, struct c_array *b, ssize_t dim, ssize_t index_a, ssize_t index_b) {
    // to be initially called with deepcopy_64bit(a, b, 0, 0, 0)
    if (dim > b->ndim - 1)
        return 0;
    ssize_t N_a = a->strides[dim] / a->itemsize;
    ssize_t N_b = b->strides[dim] / b->itemsize;
    if (dim == b->ndim - 1) {
        for (ssize_t i = 0; i < b->shape[dim]; i++)
            ((int64_t*)a->ptr)[index_a + i * N_a] = ((int64_t*)b->ptr)[index_b + i * N_b];
        return 1;
    }
    for (ssize_t i = 0; i < b->shape[dim]; i++)
        deepcopy_64bit(a, b, dim + 1, index_a + i * N_a, index_b + i * N_b);
    return 1;
}

int deepcopy_CPU(struct c_array *a, struct c_array *b) {
    /* PURPOSE:
        - deepcopies all contents from b to a
    PREREQUISITES:
        - a and b have the same size/shape/dimensionality
        - a and b are of the same type
    */
    if (b->itemsize == 1)
        return deepcopy_8bit(a, b, 0, 0, 0);
    if (b->itemsize == 2)
        return deepcopy_16bit(a, b, 0, 0, 0);
    if (b->itemsize == 4)
        return deepcopy_32bit(a, b, 0, 0, 0);
    if (b->itemsize == 8)
        return deepcopy_64bit(a, b, 0, 0, 0);
    return 0;
}

#endif                       // end include guard

#ifdef __cplusplus           // end C compiler instruction
}
#endif
