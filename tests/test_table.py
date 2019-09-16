#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import unittest

import numpy

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_table_nbytes(self):
        assert isinstance(Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).nbytes, int)

    def test_table_get(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"

        assert a.tolist() == [{"0": 0, "1": 0.0}, {"0": 1, "1": 1.1}, {"0": 2, "1": 2.2}, {"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}, {"0": 5, "1": 5.5}, {"0": 6, "1": 6.6}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}, {"0": 9, "1": 9.9}]
        assert a["0"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a["1"].tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert a[["0"]].tolist() == [{"0": 0}, {"0": 1}, {"0": 2}, {"0": 3}, {"0": 4}, {"0": 5}, {"0": 6}, {"0": 7}, {"0": 8}, {"0": 9}]

        assert a[5]["0"] == 5
        assert a["0"][5] == 5

        assert a[5]["1"] == 5.5
        assert a["1"][5] == 5.5

        assert a[5:]["0"][0] == 5
        assert a["0"][5:][0] == 5
        assert a[5:][0]["0"] == 5

        assert a[::-2]["0"][-1] == 1
        assert a["0"][::-2][-1] == 1
        assert a[::-2][-1]["0"] == 1

        assert a[[5, 3, 7, 5]]["0"].tolist() == [5, 3, 7, 5]
        assert a["0"][[5, 3, 7, 5]].tolist() == [5, 3, 7, 5]

        assert a["0"][[True, False, True, False, True, False, True, False, True, False]].tolist() == [0, 2, 4, 6, 8]
        assert a[[True, False, True, False, True, False, True, False, True, False]]["0"].tolist() == [0, 2, 4, 6, 8]

    def test_table_set(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        a["stuff"] = [5, 4, 3, 2, 1]
        assert a.tolist() == [{"0": 0, "1": 0.0, "stuff": 5}, {"0": 1, "1": 1.1, "stuff": 4}, {"0": 2, "1": 2.2, "stuff": 3}, {"0": 3, "1": 3.3, "stuff": 2}, {"0": 4, "1": 4.4, "stuff": 1}]
        a[["x", "y"]] = Table(range(3), range(100))
        assert a.tolist() == [{"0": 0, "1": 0.0, "stuff": 5, "x": 0, "y": 0}, {"0": 1, "1": 1.1, "stuff": 4, "x": 1, "y": 1}, {"0": 2, "1": 2.2, "stuff": 3, "x": 2, "y": 2}]

    def test_table_ufunc(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        b = a + 100
        assert b.tolist() == [{"0": 100, "1": 100.0}, {"0": 101, "1": 101.1}, {"0": 102, "1": 102.2}, {"0": 103, "1": 103.3}, {"0": 104, "1": 104.4}, {"0": 105, "1": 105.5}, {"0": 106, "1": 106.6}, {"0": 107, "1": 107.7}, {"0": 108, "1": 108.8}, {"0": 109, "1": 109.9}]
        c = a + b
        assert c.tolist() == [{"0": 100, "1": 100.0}, {"0": 102, "1": 102.19999999999999}, {"0": 104, "1": 104.4}, {"0": 106, "1": 106.6}, {"0": 108, "1": 108.80000000000001}, {"0": 110, "1": 111.0}, {"0": 112, "1": 113.19999999999999}, {"0": 114, "1": 115.4}, {"0": 116, "1": 117.6}, {"0": 118, "1": 119.80000000000001}]

    def test_table_slice_slice(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[::2][2:4].tolist() == [{"0": 4, "1": 4.4}, {"0": 6, "1": 6.6}]
        assert a[::2][2:4][1].tolist() == {"0": 6, "1": 6.6}
        assert a[1::2][2:100].tolist() == [{"0": 5, "1": 5.5}, {"0": 7, "1": 7.7}, {"0": 9, "1": 9.9}]
        assert a[1::2][2:100][1].tolist() == {"0": 7, "1": 7.7}
        assert a[5:][2:4].tolist() == [{"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}]
        assert a[5:][2:4][1].tolist() == {"0": 8, "1": 8.8}
        assert a[-5:][2:4].tolist() == [{"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}]
        assert a[-5:][2:4][1].tolist() == {"0": 8, "1": 8.8}
        assert a[::2][-4:-2].tolist() == [{"0": 2, "1": 2.2}, {"0": 4, "1": 4.4}]
        assert a[::2][-4:-2][1].tolist() == {"0": 4, "1": 4.4}
        assert a[::-2][2:4].tolist() == [{"0": 5, "1": 5.5}, {"0": 3, "1": 3.3}]
        assert a[::-2][2:4][1].tolist() == {"0": 3, "1": 3.3}
        assert a[::-2][2:100].tolist() == [{"0": 5, "1": 5.5}, {"0": 3, "1": 3.3}, {"0": 1, "1": 1.1}]
        assert a[::-2][2:100][1].tolist() == {"0": 3, "1": 3.3}
        assert a[::-2][3::-1].tolist() == [{"0": 3, "1": 3.3}, {"0": 5, "1": 5.5}, {"0": 7, "1": 7.7}, {"0": 9, "1": 9.9}]
        assert a[::-2][3::-1][-1].tolist() == {"0": 9, "1": 9.9}

    def test_table_slice_fancy(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[::2][[4, 3, 1, 1]].tolist() == [{"0": 8, "1": 8.8}, {"0": 6, "1": 6.6}, {"0": 2, "1": 2.2}, {"0": 2, "1": 2.2}]
        assert a[::2][[4, 3, 1, 1]][1].tolist() == {"0": 6, "1": 6.6}
        assert a[-5::-1][[0, 1, 5]].tolist() == [{"0": 5, "1": 5.5}, {"0": 4, "1": 4.4}, {"0": 0, "1": 0.0}]
        assert a[-5::-1][[0, 1, 5]][1].tolist() == {"0": 4, "1": 4.4}

    def test_table_slice_mask(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[::2][[True, False, False, True, True]].tolist() == [{"0": 0, "1": 0.0}, {"0": 6, "1": 6.6}, {"0": 8, "1": 8.8}]
        assert a[::2][[True, False, False, True, True]][1].tolist() == {"0": 6, "1": 6.6}
        assert a[-5::-1][[True, True, False, False, False, True]].tolist() == [{"0": 5, "1": 5.5}, {"0": 4, "1": 4.4}, {"0": 0, "1": 0.0}]
        assert a[-5::-1][[True, True, False, False, False, True]][1].tolist() == {"0": 4, "1": 4.4}

    def test_table_fancy_slice(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[[7, 3, 3, 4, 4][2:]].tolist() == [{"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}, {"0": 4, "1": 4.4}]
        assert a[[7, 3, 3, 4, 4][2:]][1].tolist() == {"0": 4, "1": 4.4}

    def test_table_fancy_fancy(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[[7, 3, 3, 4, 4]][[-2, 2, 0]].tolist() == [{"0": 4, "1": 4.4}, {"0": 3, "1": 3.3}, {"0": 7, "1": 7.7}]
        assert a[[7, 3, 3, 4, 4]][[-2, 2, 0]][1].tolist() == {"0": 3, "1": 3.3}

    def test_table_fancy_mask(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[[7, 3, 3, 4, 4]][[True, False, True, False, True]].tolist() == [{"0": 7, "1": 7.7}, {"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}]
        assert a[[7, 3, 3, 4, 4]][[True, False, True, False, True]][1].tolist() == {"0": 3, "1": 3.3}

    def test_table_mask_slice(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[[True, True, False, False, False, False, False, True, True, True]][1:4].tolist() == [{"0": 1, "1": 1.1}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}]
        assert a[[True, True, False, False, False, False, False, True, True, True]][1:4][1].tolist() == {"0": 7, "1": 7.7}

    def test_table_mask_fancy(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[[True, True, False, False, False, False, False, True, True, True]][[1, 2, 3]].tolist() == [{"0": 1, "1": 1.1}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}]
        assert a[[True, True, False, False, False, False, False, True, True, True]][[1, 2, 3]][1].tolist() == {"0": 7, "1": 7.7}

    def test_table_mask_mask(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert a[[True, True, False, False, False, False, False, True, True, True]][[False, True, True, True, False]].tolist() == [{"0": 1, "1": 1.1}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}]
        assert a[[True, True, False, False, False, False, False, True, True, True]][[False, True, True, True, False]][1].tolist() == {"0": 7, "1": 7.7}

    def test_indexed_table(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert IndexedArray([5, 3, 7, 5], a)["1"].tolist() == [5.5, 3.3, 7.7, 5.5]

    def test_masked_table(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert MaskedArray([False, True, True, True, True, False, False, False, False, True], a, maskedwhen=False)["1"].tolist() == [None, 1.1, 2.2, 3.3, 4.4, None, None, None, None, 9.9]

    def test_jagged_table(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.rowname = "Row"
        assert JaggedArray.fromoffsets([0, 3, 5, 5, 10], a).tolist() == [[{"0": 0, "1": 0.0}, {"0": 1, "1": 1.1}, {"0": 2, "1": 2.2}], [{"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}], [], [{"0": 5, "1": 5.5}, {"0": 6, "1": 6.6}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}, {"0": 9, "1": 9.9}]]
        assert JaggedArray.fromoffsets([0, 3, 5, 5, 10], a)["1"].tolist() == [[0.0, 1.1, 2.2], [3.3, 4.4], [], [5.5, 6.6, 7.7, 8.8, 9.9]]

    def test_chunked_table(self):
        a = Table([0, 1, 2, 3], [0.0, 1.1, 2.2, 3.3])
        a.rowname = "Row"
        b = Table([4, 5, 6, 7, 8, 9], [4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        b.rowname = "Row"
        c = ChunkedArray([a, b])
        assert c["1"][6] == 6.6

    def test_virtual_table(self):
        a = VirtualArray(lambda: Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        a.array.rowname = "Row"
        assert a.tolist() == [{"0": 0, "1": 0.0}, {"0": 1, "1": 1.1}, {"0": 2, "1": 2.2}, {"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}, {"0": 5, "1": 5.5}, {"0": 6, "1": 6.6}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}, {"0": 9, "1": 9.9}]

    def test_table_concatenate(self):
        a = Table([0, 1, 2], [0.0, 1.1, 2.2])
        b = Table([4, 5], [4.4, 5.5])
        a.rowname = "Row"
        b.rowname = "Row"

        c = a.concatenate([b])

        assert c.tolist() == [{"0": 0, "1": 0.0}, {"0": 1, "1": 1.1}, {"0": 2, "1": 2.2},
                              {"0": 4, "1": 4.4}, {"0": 5, "1": 5.5}, ]

    def test_table_unzip(self):
        left = [1, 2, 3, 4, 5]
        right = [6, 7, 8, 9, 10]
        table = Table.named("tuple", left, right)
        unzip = table.unzip()
        assert type(unzip) is tuple
        assert len(unzip) == 2
        assert all(unzip[0] == left)
        assert all(unzip[1] == right)

    def test_table_row_tuple_len(self):
        a = Table([1], [2])
        assert len(a[0]) == 2

    def test_table_row_tuple_iteration(self):
        rows = [[1, 2], [3, 4]]
        columns = zip(*rows)
        a = Table(*columns)
        b = [[element for element in row] for row in a]
        for row in b:
            for element in row:
                with self.assertRaises(TypeError, msg='Scalar row element should not have a length'):
                    len(element)
                with self.assertRaises(TypeError, msg='Scalar row element should not be iterable'):
                    iter(element)
        assert b == rows

    def test_table_row_dict_len(self):
        column_dict = {'a': [1], 'b': [2]}
        a = Table(column_dict)
        assert len(a[0]) == 2

    def test_table_row_dict_iteration(self):
        column_dict = {'a': [1, 3], 'b': [2, 4]}
        a = Table(column_dict)
        b = [{key: row[key] for key in row} for row in a]
        assert b == [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
