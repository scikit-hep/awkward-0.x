# import awkward
# import numpy
# import pyarrow
# import pandas

# arrow_buffer = pyarrow.ipc.open_file(open("tests/samples/exoplanets.arrow", "rb")).get_batch(0)
# stars = awkward.fromarrow(arrow_buffer)
# stars

# pandas_friendly = awkward.JaggedArray.zip(
#     planet_eccen = stars.planets.eccen,
#     planet_mass = stars.planets.mass,
#     planet_name = stars.planets.name,
#     planet_orbit = stars.planets.orbit,
#     planet_period = stars.planets.period,
#     planet_radius = stars.planets.radius
# )
# pandas_friendly["star_dec"] = stars.dec
# pandas_friendly["star_dist"] = stars.dist
# pandas_friendly["star_mass"] = stars.mass
# pandas_friendly["star_name"] = stars.name
# pandas_friendly["star_ra"] = stars.ra
# pandas_friendly["star_radius"] = stars.radius
# pandas_friendly["index0"] = numpy.arange(len(pandas_friendly))
# index = pandas.MultiIndex.from_arrays([pandas_friendly["index0"].flatten(), pandas_friendly.localindex.flatten()])
# columns = pandas.MultiIndex.from_tuples([
#     ("planet", "eccen"), ("planet", "mass"), ("planet", "name"), ("planet", "orbit"), ("planet", "period"), ("planet", "radius"),
#     ("dec", ""), ("dist", ""), ("mass", ""), ("name", ""), ("ra", ""), ("radius", "")])
# df = pandas.DataFrame(data={columns[i]: pandas_friendly[pandas_friendly.columns[i]].flatten() for i in range(len(columns))}, columns=columns, index=index)

# import numpy

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
            out, deferred = None, {}

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
                    raise ValueError("this array has unflattenable substructure: {0}".format(str(tpen)))

                if isinstance(tmp, awkward.array.jagged.JaggedArray):
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

            n = ""
            while n in tpe.columns:
                n = n + " "
            out[n] = numpy.arange(len(out))
            globalindex[0] = out[n].flatten()

            if any(isinstance(array[n], awkward.array.jagged.JaggedArray) for n in tpe.columns):
                localindex.insert(0, out.localindex.flatten())

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

# out = topandas_regular(stars)
