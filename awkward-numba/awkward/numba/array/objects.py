#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import awkward.array.objects
from .base import NumbaMethods

class MethodsNumba(NumbaMethods, awkward.array.objects.Methods):
    pass

class ObjectArrayNumba(NumbaMethods, awkward.array.objects.ObjectArray):
    pass

class StringArrayNumba(NumbaMethods, awkward.array.objects.StringArray):
    pass
