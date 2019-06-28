import awkward
import numpy
import pyarrow
import pandas

arrow_buffer = pyarrow.ipc.open_file(open("tests/samples/exoplanets.arrow", "rb")).get_batch(0)
stars = awkward.fromarrow(arrow_buffer)
stars

pandas_friendly = awkward.JaggedArray.zip(
    planet_eccen = stars.planets.eccen,
    planet_mass = stars.planets.mass,
    planet_name = stars.planets.name,
    planet_orbit = stars.planets.orbit,
    planet_period = stars.planets.period,
    planet_radius = stars.planets.radius
)
pandas_friendly["star_dec"] = stars.dec
pandas_friendly["star_dist"] = stars.dist
pandas_friendly["star_mass"] = stars.mass
pandas_friendly["star_name"] = stars.name
pandas_friendly["star_ra"] = stars.ra
pandas_friendly["star_radius"] = stars.radius
pandas_friendly["index0"] = numpy.arange(len(pandas_friendly))
index = pandas.MultiIndex.from_arrays([pandas_friendly["index0"].flatten(), pandas_friendly.localindex.flatten()])
columns = pandas.MultiIndex.from_tuples([
    ("planet", "eccen"), ("planet", "mass"), ("planet", "name"), ("planet", "orbit"), ("planet", "period"), ("planet", "radius"),
    ("dec", ""), ("dist", ""), ("mass", ""), ("name", ""), ("ra", ""), ("radius", "")])
df = pandas.DataFrame(data={columns[i]: pandas_friendly[pandas_friendly.columns[i]].flatten() for i in range(len(columns))}, columns=columns, index=index)

import numpy

def topandas_regular(array):
    import pandas

    import awkward.type
    import awkward.array.jagged

    globalindex = [None]
    localindex = []
    columns = []
    def recurse(array, tpe, cols):
        if isinstance(tpe, awkward.type.TableType):
            out = None
            starts, stops = None, None

            for n in tpe.columns:
                if n == "" or not isinstance(n, str):
                    raise ValueError("column names must be non-empty strings")

                tpen = tpe[n]
                colsn = cols + (n,)
                if isinstance(tpen, awkward.type.OptionType):
                    array = array.content
                    tpen = tpen.type

                if isinstance(tpen, numpy.dtype):
                    columns.append(colsn)

                elif isinstance(tpen, type) and issubclass(tpen, (str, bytes)):
                    columns.append(colsn)

                elif isinstance(tpen, awkward.type.ArrayType) and tpen.takes == numpy.inf:
                    tmp = recurse(array[n].content, tpen.to, colsn)
                    if out is None:
                        starts, stops = array[n].starts, array[n].stops
                        out = array.JaggedArray(starts, stops, array.Table({n: tmp}))
                    elif not numpy.array_equal(starts, array[n].starts) or not numpy.array_equal(stops, array[n].stops):
                        raise ValueError("this array has more than one jagged array structure")
                    else:
                        out[n] = array.JaggedArray(starts, stops, tmp)

                elif isinstance(tpen, awkward.type.TableType):
                    out[n] = recurse(array[n], tpen, colsn)

                else:
                    raise ValueError("this array has unflattenable substructure: {0}".format(str(tpen)))

            if out is None:
                out = array.Table()

            for n in tpe.columns:
                if not (isinstance(tpe[n], awkward.type.ArrayType) and tpe[n].takes == numpy.inf):
                    out[n] = array[n]

            if isinstance(out, awkward.array.jagged.JaggedArray):
                out[""] = array.numpy.arange(len(out))
                globalindex[0] = out[""].flatten()
                localindex.insert(0, out.localindex.flatten())
                out = out.flatten()
            else:
                globalindex[0] = array.numpy.arange(len(out))

            return out[tpe.columns]

        else:
            raise ValueError("this array is not a Table")

    tmp = recurse(array, awkward.type.fromarray(array).to, ())
    deepest = max(len(x) for x in columns)

    out = {}
    for i, col in enumerate(columns):
        x = tmp[col[0]]
        for c in col[1:]:
            x = x[c]
        columns[i] = col + ("",) * (deepest - len(col))
        out[columns[i]] = x

    return pandas.DataFrame(data=out,
                            index=pandas.MultiIndex.from_arrays(globalindex + localindex),
                            columns=pandas.MultiIndex.from_tuples(columns))

out = topandas_regular(stars)
