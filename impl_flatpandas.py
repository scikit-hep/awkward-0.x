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
    import pandas

    import awkward.type
    import awkward.array.jagged
    import awkward.array.table

    globalindex = [None]
    localindex = []
    columns = []
    def recurse(array, tpe, cols, seriously):
        if isinstance(tpe, awkward.type.TableType):
            out = None
            starts, stops = None, None

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

                elif isinstance(tpen, type) and issubclass(tpen, (str, bytes)):
                    columns.append(colsn)

                elif isinstance(tpen, awkward.type.ArrayType) and tpen.takes == numpy.inf:
                    print("jagged")

                    tmp = recurse(array[n].content, tpen.to, colsn, True)
                    if out is None:
                        starts, stops = array[n].starts, array[n].stops
                        out = array.JaggedArray(starts, stops, array.Table({n: tmp}))
                    elif not numpy.array_equal(starts, array[n].starts) or not numpy.array_equal(stops, array[n].stops):
                        raise ValueError("this array has more than one jagged array structure")
                    else:
                        out[n] = array.JaggedArray(starts, stops, tmp)

                elif isinstance(tpen, awkward.type.TableType):
                    print("table")

                    tmp = recurse(array[n], tpen, colsn, True)
                    if out is None:
                        out = array.Table({n: tmp})
                    else:
                        out[n] = tmp

                else:
                    raise ValueError("this array has unflattenable substructure: {0}".format(str(tpen)))

            if out is None:
                out = array.Table()

            for n in tpe.columns:
                if n not in out.columns:
                    out[n] = array[n]

            if isinstance(out, awkward.array.jagged.JaggedArray):
                n = ""
                while n in tpe.columns:
                    n = n + " "
                out[n] = array.numpy.arange(len(out))
                globalindex[0] = out[n].flatten()
                localindex.insert(0, out.localindex.flatten())
                out = out.flatten()
            else:
                globalindex[0] = array.numpy.arange(len(out))

            print("returning", out[tpe.columns].tolist())

            return out[tpe.columns]

        else:
            if isinstance(array, numpy.ndarray):
                Table = awkward.array.table.Table
            else:
                Table = array.Table

            out = recurse(Table({"": array}), awkward.type.TableType(**{"": tpe}), cols, False)[""]
            print("return", out.tolist())

            return out

    tmp = recurse(array, awkward.type.fromarray(array).to, (), True)
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
        columns = pandas.MultiIndex.from_tuples(columns)

        print("data", out)
        print("index", index)
        print("columns", columns)

        return pandas.DataFrame(data=out, index=index, columns=columns)

# out = topandas_regular(stars)
