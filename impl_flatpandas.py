def topandas_regular(array):
    import numpy
    import pandas

    import awkward.type
    import awkward.array.base
    import awkward.array.jagged
    import awkward.array.table

    if isinstance(array, awkward.array.base.AwkwardArray):
        numpy = array.numpy
        JaggedArray = array.JaggedArray
        Table = array.Table
    else:
        JaggedArray = awkward.array.jagged.JaggedArray
        Table = awkward.array.table.Table

    globalindex = [None]
    localindex = []
    columns = []
    def recurse(array, tpe, cols, seriously):
        if isinstance(tpe, awkward.type.TableType):
            starts, stops = None, None
            out, deferred, unflattened = None, {}, None

            for n in tpe.columns:
                if not isinstance(n, str):
                    raise ValueError("column names must be strings")

                tpen = tpe[n]
                colsn = cols + (n,) if seriously else cols
                if isinstance(tpen, awkward.type.OptionType):
                    array = array.content
                    tpen = tpen.type

                if isinstance(tpen, numpy.dtype):
                    columns.append(colsn)
                    tmp = array[n]

                elif isinstance(tpen, type) and issubclass(tpen, (str, bytes)):
                    columns.append(colsn)
                    tmp = array[n]

                elif isinstance(tpen, awkward.type.ArrayType) and tpen.takes == numpy.inf:
                    tmp = JaggedArray(array[n].starts, array[n].stops, recurse(array[n].content, tpen.to, colsn, True))

                elif isinstance(tpen, awkward.type.TableType):
                    tmp = recurse(array[n], tpen, colsn, True)

                else:
                    raise ValueError("this array has unflattenable substructure:\n\n{0}".format(str(tpen)))

                if isinstance(tmp, awkward.array.jagged.JaggedArray):
                    if isinstance(tmp.content, awkward.array.jagged.JaggedArray):
                        unflattened = tmp
                        tmp = tmp.flatten(axis=1)

                    if starts is None:
                        starts, stops = tmp.starts, tmp.stops
                    elif not numpy.array_equal(starts, tmp.starts) or not numpy.array_equal(stops, tmp.stops):
                        raise ValueError("this array has more than one jagged array structure")
                    if out is None:
                        out = JaggedArray(starts, stops, Table({n: tmp.content}))
                    else:
                        out[n] = tmp

                else:
                    deferred[n] = tmp

            if out is None:
                out = Table()

            for n, x in deferred.items():
                out[n] = x

            m = ""
            while m in tpe.columns:
                m = m + " "
            out[m] = numpy.arange(len(out))
            globalindex[0] = out[m].flatten()

            for n in tpe.columns:
                if isinstance(array[n], awkward.array.jagged.JaggedArray):
                    if unflattened is None:
                        localindex.insert(0, out[n].localindex.flatten())
                    else:
                        oldloc = unflattened.content.localindex
                        tab = JaggedArray(oldloc.starts, oldloc.stops, Table({"oldloc": oldloc.content}))
                        tab["newloc"] = array[n].localindex.flatten()
                        localindex.insert(0, tab["newloc"].flatten())
                    break

            return out[tpe.columns]

        else:
            return recurse(Table({"": array}), awkward.type.TableType(**{"": tpe}), cols, False)[""]

    tmp = recurse(array, awkward.type.fromarray(array).to, (), True)
    if isinstance(tmp, awkward.array.jagged.JaggedArray):
        tmp = tmp.flatten()

    deepest = max(len(x) for x in columns)

    out = {}
    for i, col in enumerate(columns):
        x = tmp
        for c in col:
            x = x[c]
        columns[i] = col + ("",) * (deepest - len(col))
        out[columns[i]] = x

    index = globalindex + localindex
    if len(index) == 1:
        index = pandas.Index(index[0])
    else:
        index = pandas.MultiIndex.from_arrays(index)

    if len(columns) == 1 and deepest == 0:
        return pandas.Series(out[()], index=index)
    else:
        return pandas.DataFrame(data=out, index=index, columns=pandas.MultiIndex.from_tuples(columns))
