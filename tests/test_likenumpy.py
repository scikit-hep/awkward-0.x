#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import unittest

import numpy

import awkward

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_likenumpy_slices(self):
        print()

        np = numpy.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]])
        aw = awkward.fromiter(np)

        assert np.tolist() == aw.tolist()
        assert np[:2].tolist() == aw[:2].tolist()
        assert np[:2, :2].tolist() == aw[:2, :2].tolist()
        assert np[:2, 2].tolist() == aw[:2, 2].tolist()
        assert np[2, :2].tolist() == aw[2, :2].tolist()
        assert np[:2, [0, 1]].tolist() == aw[:2, [0, 1]].tolist()
        assert np[[0, 1], :2].tolist() == aw[[0, 1], :2].tolist()
        assert np[:2, [0, 1, 2]].tolist() == aw[:2, [0, 1, 2]].tolist()
        assert np[[0, 1, 2], :2].tolist() == aw[[0, 1, 2], :2].tolist()
        assert np[[0, 1], [0, 1]].tolist() == aw[[0, 1], [0, 1]].tolist()
        assert np[[0, 1, 2], [0, 1, 2]].tolist() == aw[[0, 1, 2], [0, 1, 2]].tolist()
        assert np[:2, [True, False, True]].tolist() == aw[:2, [True, False, True]].tolist()
        assert np[[True, False, True], :2].tolist() == aw[[True, False, True], :2].tolist()
        assert np[[True, False, True], [True, False, True]].tolist() == aw[[True, False, True], [True, False, True]].tolist()

        np = numpy.array([[[1, 10, 100], [2, 20, 200], [3, 30, 300]], [[4, 40, 400], [5, 50, 500], [6, 60, 600]], [[7, 70, 700], [8, 80, 800], [9, 90, 900]]])
        aw = awkward.fromiter(np)

        assert np.tolist() == aw.tolist()
        assert np[:2].tolist() == aw[:2].tolist()
        assert np[:2, :2].tolist() == aw[:2, :2].tolist()
        assert np[:2, 2].tolist() == aw[:2, 2].tolist()
        assert np[2, :2].tolist() == aw[2, :2].tolist()
        assert np[:2, [0, 1]].tolist() == aw[:2, [0, 1]].tolist()
        assert np[[0, 1], :2].tolist() == aw[[0, 1], :2].tolist()
        assert np[:2, [0, 1, 2]].tolist() == aw[:2, [0, 1, 2]].tolist()
        assert np[[0, 1, 2], :2].tolist() == aw[[0, 1, 2], :2].tolist()
        assert np[[0, 1], [0, 1]].tolist() == aw[[0, 1], [0, 1]].tolist()
        assert np[[0, 1, 2], [0, 1, 2]].tolist() == aw[[0, 1, 2], [0, 1, 2]].tolist()
        assert np[:2, [True, False, True]].tolist() == aw[:2, [True, False, True]].tolist()
        assert np[[True, False, True], :2].tolist() == aw[[True, False, True], :2].tolist()
        assert np[[True, False, True], [True, False, True]].tolist() == aw[[True, False, True], [True, False, True]].tolist()

        assert np[:2, :2, 0].tolist() == aw[:2, :2, 0].tolist()
        assert np[:2, 2, 0].tolist() == aw[:2, 2, 0].tolist()
        assert np[2, :2, 0].tolist() == aw[2, :2, 0].tolist()
        assert np[:2, [0, 1], 0].tolist() == aw[:2, [0, 1], 0].tolist()
        assert np[[0, 1], :2, 0].tolist() == aw[[0, 1], :2, 0].tolist()
        assert np[:2, [0, 1, 2], 0].tolist() == aw[:2, [0, 1, 2], 0].tolist()
        assert np[[0, 1, 2], :2, 0].tolist() == aw[[0, 1, 2], :2, 0].tolist()
        assert np[[0, 1], [0, 1], 0].tolist() == aw[[0, 1], [0, 1], 0].tolist()
        assert np[[0, 1, 2], [0, 1, 2], 0].tolist() == aw[[0, 1, 2], [0, 1, 2], 0].tolist()
        assert np[:2, [True, False, True], 0].tolist() == aw[:2, [True, False, True], 0].tolist()
        assert np[[True, False, True], :2, 0].tolist() == aw[[True, False, True], :2, 0].tolist()
        assert np[[True, False, True], [True, False, True], 0].tolist() == aw[[True, False, True], [True, False, True], 0].tolist()

        assert np[:2, :2, 1].tolist() == aw[:2, :2, 1].tolist()
        assert np[:2, 2, 1].tolist() == aw[:2, 2, 1].tolist()
        assert np[2, :2, 1].tolist() == aw[2, :2, 1].tolist()
        assert np[:2, [0, 1], 1].tolist() == aw[:2, [0, 1], 1].tolist()
        assert np[[0, 1], :2, 1].tolist() == aw[[0, 1], :2, 1].tolist()
        assert np[:2, [0, 1, 2], 1].tolist() == aw[:2, [0, 1, 2], 1].tolist()
        assert np[[0, 1, 2], :2, 1].tolist() == aw[[0, 1, 2], :2, 1].tolist()
        assert np[[0, 1], [0, 1], 1].tolist() == aw[[0, 1], [0, 1], 1].tolist()
        assert np[[0, 1, 2], [0, 1, 2], 1].tolist() == aw[[0, 1, 2], [0, 1, 2], 1].tolist()
        assert np[:2, [True, False, True], 1].tolist() == aw[:2, [True, False, True], 1].tolist()
        assert np[[True, False, True], :2, 1].tolist() == aw[[True, False, True], :2, 1].tolist()
        assert np[[True, False, True], [True, False, True], 1].tolist() == aw[[True, False, True], [True, False, True], 1].tolist()
