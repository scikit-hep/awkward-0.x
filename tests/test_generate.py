#!/usr/bin/env python

# Copyright (c) 2018, DIANA-HEP
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest

import awkward

class Test(unittest.TestCase):
    def test_generate_runTest(self):
        pass

    def test_generate_empty(self):
        x = []
        assert awkward.fromiter(x).tolist() == x

    def test_generate_primitive(self):
        x = [False, True, True]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, 2, 3]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1.1, 2.2, 3.3]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1.1j, 2.2j, 3.3j]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, 2, 3]
        assert isinstance(awkward.fromiter(x).tolist()[0], int)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, 2, 3.3]
        assert isinstance(awkward.fromiter(x).tolist()[0], float)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_strings(self):
        x = [b"one", b"two", b"three"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", "two", "three"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_jagged(self):
        x = [[]]
        assert awkward.fromiter(x).tolist() == x

        x = [[], [], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [3.14], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[999], [], [999]]
        assert isinstance(awkward.fromiter(x).tolist()[0][0], int)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[999], [], [3.14]]
        assert isinstance(awkward.fromiter(x).tolist()[0][0], float)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_multijagged(self):
        x = [[[]]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[[]], [], [[]]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [[]], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [[], [], []], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[[3.14]]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[[]], [], [[3.14]]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[[3.14]], [], [[]]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [[1], [2], [3]], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[[3.14]], [], [[3.14]]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[[999]], [], [[999]]]
        assert isinstance(awkward.fromiter(x).tolist()[0][0][0], int)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[[999]], [], [[3.14]]]
        assert isinstance(awkward.fromiter(x).tolist()[0][0][0], float)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_table(self):
        x = [{"a": 1, "b": 1.1}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": 3.3}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": 3.3}]
        assert isinstance(awkward.fromiter(x).tolist()[0]["a"], int)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3.0, "b": 3.3}]
        assert isinstance(awkward.fromiter(x).tolist()[0]["a"], float)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}, {"a": 3, "b": "three"}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": b"one"}, {"a": 2, "b": b"two"}, {"a": 3, "b": b"three"}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": []}, {"a": 2, "b": [2.2]}, {"a": 3.0, "b": [3.3, 3.3]}]
        assert isinstance(awkward.fromiter(x).tolist()[0]["a"], float)
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": {"x": 1.1}}, {"a": 2, "b": {"x": 2.2}}, {"a": 3, "b": {"x": 3.3}}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_primitive_primitive(self):
        x = [1, 2, True]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [True, 1, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, True, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, 2, True, False]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [True, 1, 2, False]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, True, 2, False]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [False, 1, 2, True]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [False, True, 1, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [False, 1, True, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_primitive_strings(self):
        x = ["one", "two", 1]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, "one", "two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", 1, "two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, 2, "one"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, "one", 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", 1, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_primitive_jagged(self):
        x = [1, 2, []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, 2, [3.14]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, [], 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, [3.14], 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], 1, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[3.14], 1, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, [], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, [3.14], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, [], [3.14]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], 1, []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[3.14], 1, []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], 1, [3.14]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [], 1]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[3.14], [], 1]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [3.14], 1]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_primitive_table(self):
        x = [999, {"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, 999, {"a": 2, "b": 2.2}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, 999]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, 2, {"a": 999, "b": 3.14}]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [1, {"a": 999, "b": 3.14}, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 999, "b": 3.14}, 1, 2]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_strings_strings(self):
        x = ["one", b"two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [b"one", "two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_strings_jagged(self):
        x = ["one", "two", []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", "two", [3.14]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", [], "two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", [3.14], "two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], "one", "two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[3.14], "one", "two"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", [], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", [1.1], []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", [], [2.2]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = ["one", [1.1], [2.2]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], "one", []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[1.1], "one", []]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], "one", [2.2]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[1.1], "one", [2.2]]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [], "one"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[1.1], [], "one"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[], [2.2], "one"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

        x = [[1.1], [2.2], "one"]
        assert awkward.fromiter(x).tolist() == x
        x.insert(1, None)
        assert awkward.fromiter(x).tolist() == x
        x.insert(0, None)
        assert awkward.fromiter(x).tolist() == x

    def test_generate_strings_table(self):
        x = ["one", {"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, "one", {"a": 2, "b": 2.2}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, "one"]
        assert awkward.fromiter(x).tolist() == x

        x = ["one", "two", {"a": 1, "b": 1.1}]
        assert awkward.fromiter(x).tolist() == x

        x = ["one", {"a": 1, "b": 1.1}, "two"]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, "one", "two"]
        assert awkward.fromiter(x).tolist() == x

    def test_generate_jagged_jagged(self):
        x = [[[[[1]]]], [[[[2]]]]]
        assert awkward.fromiter(x).tolist() == x

        x = [[[[[1]]]], [[[[2]]]]]
        assert isinstance(awkward.fromiter(x).tolist()[0][0][0][0][0], int)

        x = [[[[[1]]]], [[[[2.2]]]]]
        assert isinstance(awkward.fromiter(x).tolist()[0][0][0][0][0], float)

        x = [[[[[1]]]], [[[[[2.2]]]]]]
        assert isinstance(awkward.fromiter(x).tolist()[0][0][0][0][0], int)

    def test_generate_jagged_table(self):
        x = [[], {"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, [], {"a": 2, "b": 2.2}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, []]
        assert awkward.fromiter(x).tolist() == x

        x = [[], [3.14], {"a": 1, "b": 1.1}]
        assert awkward.fromiter(x).tolist() == x

        x = [[], {"a": 1, "b": 1.1}, [3.14]]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, [], [3.14]]
        assert awkward.fromiter(x).tolist() == x

    def test_generate_table_table(self):
        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"x": 3, "y": 3.3}]
        assert awkward.fromiter(x).tolist() == x

    def test_generate_table_union(self):
        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": True}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": "three"}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": b"three"}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": b"three"}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": []}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": [3.14]}]
        assert awkward.fromiter(x).tolist() == x

        x = [{"a": 1, "b": 1.1}, {"a": 2, "b": 2.2}, {"a": 3, "b": {"x": 999, "y": 3.14}}]
        assert awkward.fromiter(x).tolist() == x
