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
index = pandas.MultiIndex.from_arrays([pandas_friendly["index0"].flatten(), pandas_friendly.index.flatten()])
columns = pandas.MultiIndex.from_tuples([
    ("planet", "eccen"), ("planet", "mass"), ("planet", "name"), ("planet", "orbit"), ("planet", "period"), ("planet", "radius"),
    ("dec", ""), ("dist", ""), ("mass", ""), ("name", ""), ("ra", ""), ("radius", "")])
df = pandas.DataFrame(data={columns[i]: pandas_friendly[pandas_friendly.columns[i]].flatten() for i in range(len(columns))}, columns=columns, index=index)

import numpy

def flatpandas(array):
    import awkward.type

    columns = []
    def recurse(array, tpe, cols):
        if isinstance(tpe, awkward.type.TableType):
            starts, stops = None, None
            for n in tpe.columns:
                tpen = tpe[n]
                colsn = cols + (n,)
                if isinstance(tpen, awkward.type.OptionType):
                    array = array.content
                    tpen = tpen.type

                if isinstance(tpen, numpy.dtype):
                    pass
                elif isinstance(tpen, type) and issubclass(tpen, (str, bytes)):
                    pass
                elif isinstance(tpen, awkward.type.ArrayType) and tpen.takes == numpy.inf:
                    if starts is None:
                        starts, stops = array[n].starts, array[n].stops
                    elif not numpy.array_equal(starts, array[n].starts) or not numpy.array_equal(stops, array[n].stops):
                        raise ValueError("this array has more than one jagged array structure")
                    recurse(array[n].content, tpen.to, colsn)
                elif isinstance(tpen, awkward.type.TableType):
                    recurse(array[n], tpen, colsn)
                else:
                    raise ValueError("this array has unflattenable substructure: {0}".format(str(tpen)))
                columns.append(colsn)


            if starts is None:
                out = array.Table()
            else:
                out = None
                for n in tpe.columns:
                    if isinstance(tpe[n], awkward.type.ArrayType) and tpe[n].takes == numpy.inf:
                        if out is None:
                            out = array.JaggedArray(starts, stops, array.Table({n: array[n].content}))
                        else:
                            out.content[n] = array.content[n]

            for n in tpe.columns:
                if not (isinstance(tpe[n], awkward.type.ArrayType) and tpe[n].takes == numpy.inf):
                    out[n] = array[n]

            return out[tpe.columns]

        else:
            raise ValueError("this array is not a Table")
    
    out = recurse(array, awkward.type.fromarray(array).to, ())
    deepest = max(len(x) for x in columns)
    for i, x in enumerate(columns):
        columns[i] = columns[i] + ("",) * (deepest - len(x))

    return out

out = flatpandas(stars)
