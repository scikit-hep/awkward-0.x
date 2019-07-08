#!/usr/bin/env python

import pandas
import awkward
import pyarrow

exoplanets = pandas.read_csv("/home/pivarski/talks/2019-05-28-lpchats-numpy-uproot-awkward/data/nasa-exoplanets.csv")
exoplanets.index = pandas.MultiIndex.from_arrays([exoplanets["pl_hostname"], exoplanets["pl_letter"]])
exoplanets.index.names = ["star", "planet"]

df = exoplanets[["ra", "dec", "st_dist", "st_mass", "st_rad", "pl_orbsmax", "pl_orbeccen", "pl_orbper", "pl_bmassj", "pl_radj"]]
df.columns = pandas.MultiIndex.from_arrays([["star"] * 5 + ["planet"] * 5,
    ["right asc. (deg)", "declination (deg)", "distance (pc)", "mass (solar)", "radius (solar)", "orbit (AU)", "eccen.", "period (days)", "mass (Jupiter)", "radius (Jupiter)"]])

stardicts = []
for (starname, planetname), row in df.iterrows():
    if len(stardicts) == 0 or stardicts[-1]["name"] != starname:
        stardicts.append({"name": starname,
                          "ra": row["star", "right asc. (deg)"],
                          "dec": row["star", "declination (deg)"],
                          "dist": row["star", "distance (pc)"],
                          "mass": row["star", "mass (solar)"],
                          "radius": row["star", "radius (solar)"],
                          "planets": []})
    stardicts[-1]["planets"].append({"name": planetname,
                                     "orbit": row["planet", "orbit (AU)"],
                                     "eccen": row["planet", "eccen."],
                                     "period": row["planet", "period (days)"],
                                     "mass": row["planet", "mass (Jupiter)"],
                                     "radius": row["planet", "radius (Jupiter)"]})

stars = awkward.fromiter(stardicts)
arrowstars = awkward.toarrow(stars)

with open("exoplanets.arrow", "wb") as sink:
    writer = pyarrow.RecordBatchFileWriter(sink, arrowstars.schema)
    for batch in arrowstars.to_batches():
        writer.write_batch(batch)
    writer.close()

# to read it back again:
awkward.fromarrow(pyarrow.ipc.open_file(open("exoplanets.arrow", "rb")).get_batch(0))
