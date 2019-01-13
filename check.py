import numba



    def _argminmax_general(self, ismin):
        if len(self._content.shape) != 1:
            raise ValueError("cannot compute arg{0} because content is not one-dimensional".format("min" if ismin else "max"))

        if ismin:
            optimum = awkward.util.numpy.argmin
        else:
            optimum = awkward.util.numpy.argmax

        out = awkward.util.numpy.empty(self._starts.shape + self._content.shape[1:], dtype=self.INDEXTYPE)

        flatout = out.reshape((-1,) + self._content.shape[1:])
        flatstarts = self._starts.reshape(-1)
        flatstops = self._stops.reshape(-1)

        content = self._content
        for i, flatstart in enumerate(flatstarts):
            flatstop = flatstops[i]
            if flatstart != flatstop:
                flatout[i] = optimum(content[flatstart:flatstop], axis=0)

        newstarts = awkward.util.numpy.arange(len(flatstarts), dtype=self.INDEXTYPE).reshape(self._starts.shape)
        newstops = awkward.util.numpy.array(newstarts)
        newstops.reshape(-1)[flatstarts != flatstops] += 1
        return self.copy(starts=newstarts, stops=newstops, content=flatout)


@numba.jit(nopython=True)
def enumerate(it, start=0):
    count = start
    for elem in it:
        yield (count, elem)
        count += 1