import time

import numpy
import numba

import awkward
import awkward.numba

num_muons = numpy.random.poisson(1.5, 100000)
pt_muons  = awkward.JaggedArray.fromcounts(num_muons, numpy.random.exponential(10, num_muons.sum()) + 50)
eta_muons = awkward.JaggedArray.fromcounts(num_muons, numpy.random.normal(0, 1, num_muons.sum()))
phi_muons = awkward.JaggedArray.fromcounts(num_muons, numpy.random.uniform(-numpy.pi, numpy.pi, num_muons.sum()))

num_jets = numpy.random.poisson(3.5, 100000)
pt_jets  = awkward.JaggedArray.fromcounts(num_jets, numpy.random.exponential(10, num_jets.sum()) + 50)
eta_jets = awkward.JaggedArray.fromcounts(num_jets, numpy.random.normal(0, 1, num_jets.sum()))
phi_jets = awkward.JaggedArray.fromcounts(num_jets, numpy.random.uniform(-numpy.pi, numpy.pi, num_jets.sum()))

####################################################### for-loopy

def run_python(num_muons, pt_muons, eta_muons, phi_muons, num_jets, pt_jets, eta_jets, phi_jets):
    offsets = numpy.empty(len(num_muons) + 1, numpy.int64)
    content = numpy.empty((num_muons * num_jets).sum())
    offsets[0] = 0
    for i in range(len(num_muons)):
        offsets[i + 1] = offsets[i]
        for muoni in range(num_muons[i]):
            pt1  = pt_muons[i][muoni]
            eta1 = eta_muons[i][muoni]
            phi1 = phi_muons[i][muoni]
            for jeti in range(num_jets[i]):
                pt2  = pt_jets[i][jeti]
                eta2 = eta_jets[i][jeti]
                phi2 = phi_jets[i][jeti]
                content[offsets[i + 1]] = numpy.sqrt(2*pt1*pt2*(numpy.cosh(eta1 - eta2) - numpy.cos(phi1 - phi2)))
                offsets[i + 1] += 1
    return awkward.JaggedArray(offsets[:-1], offsets[1:], content)

starttime = time.time()
mass1 = run_python(num_muons, pt_muons, eta_muons, phi_muons, num_jets, pt_jets, eta_jets, phi_jets)
print("Python", time.time() - starttime)

####################################################### Numpythonic

def run_numpy(num_muons, pt_muons, eta_muons, phi_muons, num_jets, pt_jets, eta_jets, phi_jets):
    def unzip(pairs):
        return pairs.i0, pairs.i1

    pt1, pt2   = unzip(pt_muons.cross(pt_jets))
    eta1, eta2 = unzip(eta_muons.cross(eta_jets))
    phi1, phi2 = unzip(phi_muons.cross(phi_jets))

    return numpy.sqrt(2*pt1*pt2*(numpy.cosh(eta1 - eta2) - numpy.cos(phi1 - phi2)))

for i in range(5):
    starttime = time.time()
    mass2 = run_numpy(num_muons, pt_muons, eta_muons, phi_muons, num_jets, pt_jets, eta_jets, phi_jets)
    print("Numpy", time.time() - starttime)

####################################################### with Numba

run_numba = numba.jit(nopython=True)(run_python)

for i in range(5):
    starttime = time.time()
    mass3 = run_numba(num_muons, pt_muons, eta_muons, phi_muons, num_jets, pt_jets, eta_jets, phi_jets)
    print("Numba", time.time() - starttime)
