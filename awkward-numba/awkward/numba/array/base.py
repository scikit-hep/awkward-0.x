#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import operator

import numpy
import numba

class NumbaMethods(object):
    NUMBA_DEFAULTTYPE = numba.from_dtype(numpy.float64)
    NUMBA_CHARTYPE    = numba.from_dtype(numpy.uint8)
    NUMBA_INDEXTYPE   = numba.from_dtype(numpy.int64)
    NUMBA_TAGTYPE     = numba.from_dtype(numpy.uint8)
    NUMBA_MASKTYPE    = numba.from_dtype(numpy.bool_)
    NUMBA_BITMASKTYPE = numba.from_dtype(numpy.uint8)
    NUMBA_BOOLTYPE    = numba.from_dtype(numpy.bool_)

    @property
    def awkward(self):
        import awkward.numba
        return awkward.numba

    @property
    def ChunkedArray(self):
        import awkward.numba.array.chunked
        return awkward.numba.array.chunked.ChunkedArrayNumba

    @property
    def AppendableArray(self):
        import awkward.numba.array.chunked
        return awkward.numba.array.chunked.AppendableArrayNumba

    @property
    def IndexedArray(self):
        import awkward.numba.array.indexed
        return awkward.numba.array.indexed.IndexedArrayNumba

    @property
    def SparseArray(self):
        import awkward.numba.array.indexed
        return awkward.numba.array.indexed.SparseArrayNumba

    @property
    def JaggedArray(self):
        import awkward.numba.array.jagged
        return awkward.numba.array.jagged.JaggedArrayNumba

    @property
    def MaskedArray(self):
        import awkward.numba.array.masked
        return awkward.numba.array.masked.MaskedArrayNumba

    @property
    def BitMaskedArray(self):
        import awkward.numba.array.masked
        return awkward.numba.array.masked.BitMaskedArrayNumba

    @property
    def IndexedMaskedArray(self):
        import awkward.numba.array.masked
        return awkward.numba.array.masked.IndexedMaskedArrayNumba

    @property
    def Methods(self):
        import awkward.numba.array.objects
        return awkward.numba.array.objects.MethodsNumba

    @property
    def ObjectArray(self):
        import awkward.numba.array.objects
        return awkward.numba.array.objects.ObjectArrayNumba

    @property
    def StringArray(self):
        import awkward.numba.array.objects
        return awkward.numba.array.objects.StringArrayNumba

    @property
    def Table(self):
        import awkward.numba.array.table
        return awkward.numba.array.table.TableNumba

    @property
    def UnionArray(self):
        import awkward.numba.array.union
        return awkward.numba.array.union.UnionArrayNumba

    @property
    def VirtualArray(self):
        import awkward.numba.array.virtual
        return awkward.numba.array.virtual.VirtualArrayNumba

class AwkwardArrayType(numba.types.Type):
    pass

######################################################################## getitem

@numba.typing.templates.infer_global(len)
class _AwkwardArrayType_type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytype, = args
            if isinstance(arraytype, AwkwardArrayType):
                return numba.typing.templates.signature(numba.types.intp, arraytype)

@numba.typing.templates.infer_global(operator.getitem)
class _AwkwardArrayType_type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            arraytype, wheretype = args
            if isinstance(arraytype, AwkwardArrayType):
                original_wheretype = wheretype
                if not isinstance(wheretype, numba.types.BaseTuple):
                    wheretype = numba.types.Tuple((wheretype,))
                if len(wheretype.types) == 0:
                    return arraytype

                if any(isinstance(x, numba.types.Array) and x.ndim == 1 for x in wheretype.types):
                    wheretype = numba.types.Tuple(tuple(numba.types.Array(x, 1, "C") if isinstance(x, numba.types.Integer) else x for x in wheretype))

                return numba.typing.templates.signature(arraytype.getitem(wheretype), arraytype, original_wheretype)

######################################################################## iteration

@numba.typing.templates.infer
class _AwkwardArrayType_type_getiter(numba.typing.templates.AbstractTemplate):
    key = "getiter"
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytype, = args
            if isinstance(arraytype, AwkwardArrayType):
                return numba.typing.templates.signature(AwkwardArrayIteratorType(arraytype), arraytype)

class AwkwardArrayIteratorType(numba.types.common.SimpleIteratorType):
    def __init__(self, arraytype):
        self.arraytype = arraytype
        super(AwkwardArrayIteratorType, self).__init__("iter({0})".format(self.arraytype.name), self.arraytype.contenttype)

@numba.datamodel.registry.register_default(AwkwardArrayIteratorType)
class AwkwardArrayIteratorModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("index", numba.types.EphemeralPointer(numba.types.int64)),
                   ("array", fe_type.arraytype)]
        super(AwkwardArrayIteratorModel, self).__init__(dmm, fe_type, members)

@numba.extending.lower_builtin("getiter", AwkwardArrayType)
def _AwkwardArray_lower_getiter(context, builder, sig, args):
    arraytype, = sig.args
    arrayval, = args

    iterator = context.make_helper(builder, sig.return_type)
    iterator.index = numba.cgutils.alloca_once_value(builder, context.get_constant(numba.types.int64, 0))
    iterator.array = arrayval

    if context.enable_nrt:
        context.nrt.incref(builder, arraytype, arrayval)

    return numba.targets.imputils.impl_ret_new_ref(context, builder, sig.return_type, iterator._getvalue())

@numba.extending.lower_builtin("iternext", AwkwardArrayIteratorType)
@numba.targets.imputils.iternext_impl
def _AwkwardArray_lower_iternext(context, builder, sig, args, result):
    iteratortype, = sig.args
    iteratorval, = args

    iterator = context.make_helper(builder, iteratortype, value=iteratorval)
    array = numba.cgutils.create_struct_proxy(iteratortype.arraytype)(context, builder, value=iterator.array)

    index = builder.load(iterator.index)
    is_valid = builder.icmp_signed("<", index, iteratortype.arraytype.len_impl(context, builder, numba.types.intp(iteratortype.arraytype), (iterator.array,)))
    result.set_valid(is_valid)

    with builder.if_then(is_valid, likely=True):
        result.yield_(iteratortype.arraytype.getitem_impl(context, builder, iteratortype.yield_type(iteratortype.arraytype, numba.types.int64), (iterator.array, index)))
        nextindex = numba.cgutils.increment_index(builder, index)
        builder.store(nextindex, iterator.index)

######################################################################## utilities

def clsrepr(cls):
    import awkward.array
    if any(cls is x for x in (awkward.array.base.AwkwardArray,
                              awkward.array.chunked.ChunkedArray,
                              awkward.array.chunked.AppendableArray,
                              awkward.array.indexed.IndexedArray,
                              awkward.array.indexed.SparseArray,
                              awkward.array.jagged.JaggedArray,
                              awkward.array.masked.MaskedArray,
                              awkward.array.masked.BitMaskedArray,
                              awkward.array.masked.IndexedMaskedArray,
                              awkward.array.objects.Methods,
                              awkward.array.objects.ObjectArray,
                              awkward.array.objects.StringArray,
                              awkward.array.table.Table,
                              awkward.array.union.UnionArray,
                              awkward.array.virtual.VirtualArray)):
        return cls.__name__
    else:
        bases = ", ".join(clsrepr(x) for x in cls.__bases__ if x is not object)
        if len(bases) != 0:
            bases = "<" + bases + ">"
        return "{0}.{1}{2}".format(cls.__module__, cls.__name__, bases)

def sliceval2(context, builder, start, stop):
    out = context.make_helper(builder, numba.types.slice2_type)
    out.start = start
    out.stop = stop
    out.step = context.get_constant(numba.types.intp, 1)
    return out._getvalue()

def sliceval3(context, builder, start, stop, step):
    out = context.make_helper(builder, numba.types.slice3_type)
    out.start = start
    out.stop = stop
    out.step = step
    return out._getvalue()

ISADVANCED = numba.types.Array(numba.types.int64, 1, "C")
NOTADVANCED = numba.types.none
