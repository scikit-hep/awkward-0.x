#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import unittest

import pytest
import numpy

import awkward

uproot_methods = pytest.importorskip("uproot_methods")

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_physics_jetcleaning(self):
        jet_m   = awkward.fromiter([[60.0, 70.0, 80.0],       [],     [90.0, 100.0]])

        jet_pt  = awkward.fromiter([[10.0, 20.0, 30.0],       [],     [40.0, 50.0]])
        e_pt    = awkward.fromiter([[20.2, 50.5],             [50.5], [50.5]])

        jet_eta = awkward.fromiter([[-3.0, -2.0, 2.0],        [],     [-1.0, 1.0]])
        e_eta   = awkward.fromiter([[-2.2, 0.0],              [0.0],  [1.1]])

        jet_phi = awkward.fromiter([[-1.5,  0.0, 1.5],        [],     [0.78, -0.78]])
        e_phi   = awkward.fromiter([[ 0.1, 0.78],             [0.78], [-0.77]])

        jets      = uproot_methods.TLorentzVectorArray.from_ptetaphim(jet_pt, jet_eta, jet_phi, jet_m)
        electrons = uproot_methods.TLorentzVectorArray.from_ptetaphim(e_pt, e_eta, e_phi, 0.000511)

        combinations = jets.cross(electrons, nested=True)

        def delta_r(one, two):
            return one.delta_r(two)

        assert (~(delta_r(combinations.i0, combinations.i1) < 0.5).any()).tolist() == [[True, False, True], [], [True, False]]

        (jets[~(delta_r(combinations.i0, combinations.i1) < 0.5).any()])

    def test_physics_matching(self):
        gen_pt   = awkward.fromiter([[10.0, 20.0, 30.0],       [],     [40.0, 50.0]])
        reco_pt  = awkward.fromiter([[20.2, 10.1, 30.3, 50.5], [50.5], [50.5]])

        gen_eta  = awkward.fromiter([[-3.0, -2.0, 2.0],        [],     [-1.0, 1.0]])
        reco_eta = awkward.fromiter([[-2.2, -3.3, 2.2, 0.0],   [0.0],  [1.1]])

        gen_phi  = awkward.fromiter([[-1.5,  0.0, 1.5],        [],     [0.78, -0.78]])
        reco_phi = awkward.fromiter([[ 0.1, -1.4, 1.4, 0.78],  [0.78], [-0.77]])

        gen  = uproot_methods.TLorentzVectorArray.from_ptetaphim(gen_pt, gen_eta, gen_phi, 0.2)
        reco = uproot_methods.TLorentzVectorArray.from_ptetaphim(reco_pt, reco_eta, reco_phi, 0.2)

        ("gen", gen)
        ("reco", reco)

        ("gen.cross(reco)", gen.cross(reco))

        pairing = gen.cross(reco, nested=True)
        ("pairing = gen.cross(reco, nested=True)", gen.cross(reco, nested=True))

        metric = pairing.i0.delta_r(pairing.i1)
        ("metric = pairing.i0.delta_r(pairing.i1)", metric)

        index_of_minimized = metric.argmin()
        ("index_of_minimized = metric.argmin()", index_of_minimized)
        assert index_of_minimized.tolist() == [[[1], [0], [2]], [], [[0], [0]]]

        ("metric[index_of_minimized]", metric[index_of_minimized])

        passes_cut = (metric[index_of_minimized] < 0.5)
        ("passes_cut = (metric[index_of_minimized] < 0.5)", passes_cut)
        assert passes_cut.tolist() == [[[True], [True], [True]], [], [[False], [True]]]

        best_pairings_that_pass_cut = pairing[index_of_minimized][passes_cut]
        ("best_pairings_that_pass_cut = pairing[index_of_minimized][passes_cut]", best_pairings_that_pass_cut)

        genrecos = best_pairings_that_pass_cut.flatten(axis=1)
        ("genrecos = best_pairings_that_pass_cut.flatten(axis=1)", genrecos)

        ("genrecos.counts", genrecos.counts)
        ("gen.counts", gen.counts)
        assert genrecos.counts.tolist() == [3, 0, 1]
        assert gen.counts.tolist() == [3, 0, 2]

        ("genrecos.i0.pt", genrecos.i0.pt)
        ("genrecos.i1.pt", genrecos.i1.pt)

    def test_jagged_tlorentzvectorarray_concatenate(self):

        def make_jets_and_electrons():
            jet_m   = awkward.fromiter([[60.0, 70.0, 80.0],       [],     [90.0, 100.0]])

            jet_pt  = awkward.fromiter([[10.0, 20.0, 30.0],       [],     [40.0, 50.0]])
            e_pt    = awkward.fromiter([[20.2, 50.5],             [50.5], [50.5]])

            jet_eta = awkward.fromiter([[-3.0, -2.0, 2.0],        [],     [-1.0, 1.0]])
            e_eta   = awkward.fromiter([[-2.2, 0.0],              [0.0],  [1.1]])

            jet_phi = awkward.fromiter([[-1.5,  0.0, 1.5],        [],     [0.78, -0.78]])
            e_phi   = awkward.fromiter([[ 0.1, 0.78],             [0.78], [-0.77]])

            jets      = uproot_methods.TLorentzVectorArray.from_ptetaphim(jet_pt, jet_eta, jet_phi, jet_m)
            electrons = uproot_methods.TLorentzVectorArray.from_ptetaphim(e_pt, e_eta, e_phi, 0.000511)

            return jets, electrons

        jets0, electrons0 = make_jets_and_electrons()
        jets1, electrons1 = make_jets_and_electrons()
        jets2, electrons2 = make_jets_and_electrons()

        def delta_r(one, two):
            return one.delta_r(two)

        jets01 = jets0.concatenate([jets1], axis=1)
        electrons01 = electrons0.concatenate([electrons1], axis=1)

        jets = jets01.concatenate([jets2], axis=0)
        electrons = electrons01.concatenate([electrons2], axis=0)

        combinations = jets.cross(electrons, nested=True)

        reflist = [[True, False, True, True, False, True], [], [True, False, True, False],
                   [True, False, True], [], [True, False]]
        assert (~(delta_r(combinations.i0, combinations.i1) < 0.5).any()).tolist() == reflist


if __name__ == "__main__":

    unittest.main(verbosity=2)
