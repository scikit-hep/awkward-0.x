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
        super(DependentType, self).__init__(name="DependentType()")

    @property
    def name(self):
        return "DependentType({0})".format(", ".join(sorted(self.requested)))

    @name.setter
    def name(self, value):
        pass

    def request(self, name):
        if name not in self.requested:
            self.requested.add(name)
            raise TypeError("try again")

@numba.typing.templates.infer
class _DependentType_type_getitem(numba.typing.templates.AbstractTemplate):
    key = "static_getitem"
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            objtype, where = args
            print("BEFORE", objtype.requested)
            objtype.request(where)
            print("AFTER", objtype.requested)

@numba.extending.register_model(DependentType)
class DependentModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = []
        super(DependentModel, self).__init__(dmm, fe_type, members)

@numba.njit
def f(x):
    x["one"]

f(Dependent(one=1, two=2, three=3))
