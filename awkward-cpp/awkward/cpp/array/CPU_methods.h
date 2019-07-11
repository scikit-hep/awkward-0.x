#ifdef __cplusplus           // specifies that this is C, not C++
extern "C" {
#endif

#ifndef CPU_METHODS_H        // include guard
#define CPU_METHODS_H

#include <stdint.h>
#include <complex.h>

struct C_array_8 {           // single-dimensional?
    int8_t  *ptr      = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    const char *format = 0; // '=', '<', or '>'
    ssize_t strides   = 0;
};

struct C_array_16 {
    int16_t *ptr      = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    const char *format = 0;
    ssize_t strides   = 0;
};

struct C_array_32 {
    int32_t *ptr      = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    const char *format = 0;
    ssize_t strides   = 0;
};

struct C_array_64 {
    int64_t *ptr      = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    const char *format = 0;
    ssize_t strides   = 0;
};

int byteswap_16bit(int16_t *val) {
    return (*val = *val << 8 | *val >> 8) ? 1 : 0;
}

int byteswap_32bit(int32_t *val) {
    *val = ((*val << 8) & 0xFF00FF00) | ((*val >> 8) & 0xFF00FF);
    return (*val = (*val << 16) | (*val >> 16)) ? 1 : 0;
}

int byteswap_64bit(int64_t *val) {
    *val = ((*val << 8) & 0xFF00FF00FF00FF00ULL) | ((*val >> 8) & 0x00FF00FF00FF00FFULL);
    *val = ((*val << 16) & 0xFFFF0000FFFF0000ULL) | ((*val >> 16) & 0x0000FFFF0000FFFFULL);
    return (*val = (*val << 32) | (*val >> 32)) ? 1 : 0;
}

int isNative(char input) {   // returns true if native, false if non-native
    if (input == '=')
        return 1;
    union {
        unsigned long int i;
        char c[4];
    } bint = { 0x01020304 };
    return ((bint.c[0] == 1 && input != '<')
        || (bint.c[0] != 1 && input != '>'));
}

int isInt(const char *format) {
    char *intList = "qQlLhHbB";
    for (ssize_t i = 0; i < 8; i++)
        if (intList[i] == format[0] || intList[i] == format[1])
            return 1;
    return 0;
}

int makeNative_16bit(struct C_array_16 *input) {
    int N = input->strides / input->itemsize;
    if (!isNative(input->format[0]))
        for (ssize_t i = 0; i < input->size; i++)
            if (!byteswap_16bit(&(input->ptr[i * N])))
                return 0;
    return 1;
}

int makeNative_32bit(struct C_array_32 *input) {
    int N = input->strides / input->itemsize;
    if (!isNative(input->format[0]))
        for (ssize_t i = 0; i < input->size; i++)
            if (!byteswap_32bit(&(input->ptr[i * N])))
                return 0;
    return 1;
}

int makeNative_64bit(struct C_array_64 *input) {
    int N = input->strides / input->itemsize;
    if (!isNative(input->format[0]))
        for (ssize_t i = 0; i < input->size; i++)
            if (!byteswap_64bit(&(input->ptr[i * N])))
                return 0;
    return 1;
}

int checkunsigned2signed_8bit(struct C_array_8 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if ((input->ptr[i * N]) >> 7)
            return 0;
    return 1;
}

int checkunsigned2signed_16bit(struct C_array_16 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if ((input->ptr[i * N]) >> 7)
            return 0;
    return 1;
}

int checkunsigned2signed_32bit(struct C_array_32 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if ((input->ptr[i * N]) >> 7)
            return 0;
    return 1;
}

int checkunsigned2signed_64bit(struct C_array_64 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if ((input->ptr[i * N]) >> 7)
            return 0;
    return 1;
}

int checkPos_8bit(struct C_array_8 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if (input->ptr[i * N] < 0)
            return 0;
    return 1;
}

int checkPos_16bit(struct C_array_16 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if (input->ptr[i * N] < 0)
            return 0;
    return 1;
}

int checkPos_32bit(struct C_array_32 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if (input->ptr[i * N] < 0)
            return 0;
    return 1;
}

int checkPos_64bit(struct C_array_64 *input) {
    int N = input->strides / input->itemsize;
    for (ssize_t i = 0; i < input->size; i++)
        if (input->ptr[i * N] < 0)
            return 0;
    return 1;
}

int offsets2parents_int64(struct C_array_64 *offsets, struct C_array_64 *parents) {
    if (!makeNative_64bit(offsets))
        return 0;
    int N_off = offsets->strides / offsets->itemsize;
    int N_par = parents->strides / parents->itemsize;
    ssize_t j = 0;
    ssize_t k = -1;
    for (ssize_t i = 0; i < offsets->size; i++) {
        while (j < offsets->ptr[i * N_off])
            parents->ptr[N_par * j++] = (int64_t)k;
        k++;
    }
    return 1;
}

int offsets2parents_int32(struct C_array_32 *offsets, struct C_array_32 *parents) {
    if (!makeNative_32bit(offsets))
        return 0;
    int N_off = offsets->strides / offsets->itemsize;
    int N_par = parents->strides / parents->itemsize;
    ssize_t j = 0;
    ssize_t k = -1;
    for (ssize_t i = 0; i < offsets->size; i++) {
        while (j < offsets->ptr[i * N_off])
            parents->ptr[N_par * j++] = (int32_t)k;
        k++;
    }
    return 1;
}


#endif                       // end include guard

#ifdef __cplusplus           // end C compiler instruction
}
#endif
