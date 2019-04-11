import operator

import numpy
import numba

class Dependent(object):
    def __init__(self, **available):
        self.available = available

    def __getitem__(self, where):
        return self.available[where]

@numba.extending.typeof_impl.register(Dependent)
def _Dependent_typeof(val, c):
    return DependentType(list(val.available))

class DependentType(numba.types.Type):
    def __init__(self, available):
        self.available = available
        self.requested = set()
        super(DependentType, self).__init__(name="DependentType({0})".format(", ".join(sorted(self.available))))

    def request(self, name):
        if name not in self.requested:
            self.requested.add(name)

@numba.typing.templates.infer
class _DependentType_type_getitem(numba.typing.templates.AbstractTemplate):
    key = "static_getitem"
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            objtype, where = args
            print("checking type", objtype, where)
            objtype.request(where)
            return numba.types.int64

@numba.extending.register_model(DependentType)
class DependentModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        print("making model", fe_type)
        print("requested", fe_type.requested)
        members = []
        super(DependentModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(DependentType)
def _JaggedArray_unbox(typ, obj, c):
    print("unboxing", typ)
    print("requested", typ.requested)
    out = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error)

@numba.extending.lower_builtin("static_getitem", DependentType, numba.types.StringLiteral)
def _DependentType_lower_static_getitem(context, builder, sig, args):
    print("lowering", sig.args[0], sig.args[1].literal_value)
    return context.get_constant(numba.types.int64, 999)

obj = Dependent(one=999, two=999, three=999)

@numba.njit
def f(x):
    return x["one"]
print(f(obj))
print(f(obj))

@numba.njit
def g(x):
    return x["two"]
print(g(obj))
print(g(obj))
print(f(obj))

obj = Dependent(one=999, two=999, three=999)

print(f(obj))
print(g(obj))

obj = Dependent(one=999, two=999, three=999)

@numba.njit
def h(x):
    return f(x)
print(h(obj))
