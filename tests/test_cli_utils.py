import argparse
import unittest

import jsonargparse

from nn_lib.utils.cli import flatten_params


class TestCLIUtils(unittest.TestCase):
    def setUp(self):
        self.params = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        self.args = argparse.Namespace(a=1, b=2, c=argparse.Namespace(d=3, e=4))
        self.jsargs = jsonargparse.Namespace(a=1, b=2, c=jsonargparse.Namespace(d=3, e=4))

    def test_flatten_params(self):
        for typ, p in zip(
            ["dict", "argparse", "jsonargparse"], [self.params, self.args, self.jsargs]
        ):
            with self.subTest(typ):
                flattened = flatten_params(p)
                self.assertEqual(flattened, {"a": 1, "b": 2, "c/d": 3, "c/e": 4})

    def test_flatten_params_ignore_list(self):
        for typ, p in zip(
            ["dict", "argparse", "jsonargparse"], [self.params, self.args, self.jsargs]
        ):
            with self.subTest(typ):
                # Test with a high-level ignore
                flattened = flatten_params(p, ignore=["c"])
                self.assertEqual(flattened, {"a": 1, "b": 2})

    def test_flatten_params_ignore_nested(self):
        for typ, p in zip(
            ["dict", "argparse", "jsonargparse"], [self.params, self.args, self.jsargs]
        ):
            with self.subTest(typ):
                # Test with a low-level ignore
                flattened = flatten_params(p, ignore={"a": ..., "c": {"d": ...}})
                self.assertEqual(flattened, {"b": 2, "c/e": 4})
