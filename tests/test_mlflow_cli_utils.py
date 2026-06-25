import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import jsonargparse
import mlflow

from nn_lib.utils import search_single_run_by_params, log_params_and_config
from nn_lib.utils.mlflow_cli import save_as_artifact, load_artifact, flatten_params


class DummyBase(object):
    pass


class DummySubclassA(DummyBase):
    pass


class DummySubclassB(DummyBase):
    pass


class TestMLFlowUtils(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.uri = os.path.abspath(os.path.join(self.tempdir.name, "mlruns"))
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment("test_experiment")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_save_load_artifact(self):
        obj = {"hello": "world"}
        with mlflow.start_run():
            save_as_artifact(obj, "path/to/test_artifact.pkl")
            run_id = mlflow.active_run().info.run_id

        recovered_obj = load_artifact("path/to/test_artifact.pkl", run_id)

        self.assertEqual(obj, recovered_obj)

    def test_search_jsonargparse_objects(self):
        # Test that search_runs_by_params works when args are something fancy like an instantiatable
        # object spec handled by jsonargparse

        def fn_with_instantiatable_args(
            arg1: str, arg2: DummyBase, arg3: type[DummyBase] = DummySubclassA
        ):
            pass

        parser = jsonargparse.ArgumentParser()
        parser.add_function_arguments(fn_with_instantiatable_args)
        args = parser.parse_args(["--arg1", "foo", "--arg2", "DummySubclassA"])

        with mlflow.start_run():
            log_params_and_config(args, parser)
            run_id = mlflow.active_run().info.run_id
            self.assertTrue((Path(mlflow.active_run().info.artifact_uri) / "config.yaml").exists())
        the_run = search_single_run_by_params(experiment_name="test_experiment", params=args)
        self.assertEqual(the_run.info.run_id, run_id)


class TestCLIUtils(unittest.TestCase):
    def setUp(self):
        self.params = jsonargparse.Namespace(a=1, b=2, c=jsonargparse.Namespace(d=3, e=4))

    def test_flatten_params(self):
        flattened = flatten_params(self.params)
        self.assertEqual(flattened, {"a": 1, "b": 2, "c.d": 3, "c.e": 4})

    def test_flatten_params_ignore_list(self):
        # Test with a high-level skipping of a nested object
        flattened = flatten_params(self.params, skip_keys=["c"])
        self.assertEqual(flattened, {"a": 1, "b": 2})

    def test_flatten_params_ignore_list_too_deep(self):
        # Test that we don't skip if the 'key' is a prefix of the 'skip_key'
        flattened = flatten_params(self.params, skip_keys=["c.d.f"])
        self.assertEqual(flattened, {"a": 1, "b": 2, "c.d": 3, "c.e": 4})

    def test_flatten_params_ignore_nested(self):
        # Test that we can skip both high-level keys and inner-keys
        flattened = flatten_params(self.params, skip_keys=["b", "c.d"])
        self.assertEqual(flattened, {"a": 1, "c.e": 4})
