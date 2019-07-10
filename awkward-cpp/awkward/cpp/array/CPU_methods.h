#ifdef __cplusplus           // specifies that this is C, not C++
extern "C" {
#endif

#ifndef CPU_METHODS_H        // include guard
#define CPU_METHODS_H

struct C_array_8 {           // single-dimensional?
    char    *ptr      = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    char    byteorder = '='; // '=', '<', or '>'
    ssize_t strides   = 0;
};

struct C_array_16 {
    short int *ptr    = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    char    byteorder = '=';
    ssize_t strides   = 0;
};

struct C_array_32 {
    long int *ptr     = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    char    byteorder = '=';
    ssize_t strides   = 0;
};

struct C_array_64 {
    long long *ptr      = NULL;
    ssize_t itemsize  = 0;
    ssize_t size      = 0;
    char    byteorder = '=';
    ssize_t strides   = 0;
};

int byteswap_16bit(short int *val) {
    return (*val = *val << 8 | *val >> 8) ? 1 : 0;
}

int byteswap_32bit(long int *val) {
    *val = ((*val << 8) & 0xFF00FF00) | ((*val >> 8) & 0xFF00FF);
    return (*val = (*val << 16) | (*val >> 16)) ? 1 : 0;
}

int byteswap_64bit(long long *val) {
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

int makeNative_16bit(struct C_array_16 *input) {
    if (input->itemsize != 2)
        return 0;
    int N = input->strides / input->itemsize;
    if (!isNative(input->byteorder))
        for (ssize_t i = 0; i < input->size; i++)
            if (!byteswap_16bit(&(input->ptr[i * N])))
                return 0;
    return 1;
}

int makeNative_32bit(struct C_array_32 *input) {
    if (input->itemsize != 4)
        return 0;
    int N = input->strides / input->itemsize;
    if (!isNative(input->byteorder))
        for (ssize_t i = 0; i < input->size; i++)
            if (!byteswap_32bit(&(input->ptr[i * N])))
                return 0;
    return 1;
}

int makeNative_64bit(struct C_array_64 *input) {
    if (input->itemsize != 8)
        return 0;
    int N = input->strides / input->itemsize;
    if (!isNative(input->byteorder))
        for (ssize_t i = 0; i < input->size; i++)
            if (!byteswap_64bit(&(input->ptr[i * N])))
                return 0;
    return 1;
}

int offsets2parents_int64(struct C_array_64 *offsets, struct C_array_64 *parents) {
    makeNative_64bit(offsets);
    if (offsets->itemsize != 8 || parents->itemsize != 8)
        return 0;
    int N_off = offsets->strides / offsets->itemsize;
    int N_par = parents->strides / parents->itemsize;
    ssize_t j = 0;
    ssize_t k = -1;
    for (ssize_t i = 0; i < offsets->size; i++) {
        while (j < offsets->ptr[i * N_off])
            parents->ptr[N_par * j++] = (long long)k;
        k++;
    }
    return 1;
}


#endif                       // end include guard

#ifdef __cplusplus           // end C compiler instruction
}
#endif
