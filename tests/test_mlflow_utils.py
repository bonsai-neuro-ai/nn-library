import os
import unittest
from tempfile import TemporaryDirectory

import jsonargparse
import mlflow

from nn_lib.utils import log_flattened_params, search_single_run_by_params, search_runs_by_params
from nn_lib.utils.mlflow import save_as_artifact, load_artifact


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
            log_flattened_params(args)
            run_id = mlflow.active_run().info.run_id

        the_run = search_single_run_by_params(experiment_name="test_experiment", params=args)
        self.assertEqual(the_run["run_id"], run_id)
