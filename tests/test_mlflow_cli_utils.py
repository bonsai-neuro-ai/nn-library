import json
import os
import unittest
from tempfile import TemporaryDirectory

import jsonargparse
import mlflow

from nn_lib.utils import (
    search_single_run_by_params,
    open_mlflow_artifact_file,
)
from nn_lib.utils.mlflow_cli import (
    save_as_artifact,
    load_artifact,
    flatten_params,
    run_has_params,
)


class DummyBase(object):
    pass


class DummySubclassA(DummyBase):
    pass


class DummySubclassB(DummyBase):
    pass


class TestMLFlowUtils(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.uri = os.path.abspath(os.path.join("sqlite:///", self.tempdir.name, "mlflow.db"))
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

        self.assertDictEqual(obj, recovered_obj)

    def test_write_to_artifact(self):
        obj = {"hello": "world"}
        with mlflow.start_run() as run:
            with open_mlflow_artifact_file("path/to/test_dump.json", "w") as f:
                json.dump(obj, f)

        with TemporaryDirectory() as download_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path="path/to/test_dump.json",
                dst_path=download_dir,
            )
            with open(local_path, "r") as f:
                recovered_obj = json.load(f)

        self.assertDictEqual(obj, recovered_obj)

    def test_append_to_artifact(self):
        with mlflow.start_run() as run:
            with open_mlflow_artifact_file("path/to/test.txt", "w") as f:
                f.write("hello ")

            with open_mlflow_artifact_file("path/to/test.txt", "a") as f:
                f.write("world")

            restored_text = mlflow.artifacts.load_text(run.info.artifact_uri + "/path/to/test.txt")

        self.assertEqual(restored_text, "hello world")

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
            mlflow.log_params(flatten_params(args))
            run_id = mlflow.active_run().info.run_id
        the_run = search_single_run_by_params(experiment_name="test_experiment", params=args)
        self.assertEqual(the_run.info.run_id, run_id)

    def test_stored_run_has_params(self):
        params = jsonargparse.Namespace(a=1, b=2, c=jsonargparse.Namespace(d=3, e=4))
        with mlflow.start_run() as run:
            mlflow.log_params(flatten_params(params))

        run = mlflow.get_run(run_id=run.info.run_id)
        self.assertTrue(run_has_params(run, params))

        params.b = 3
        self.assertFalse(run_has_params(run, params))

    def test_stored_run_has_params_subset(self):
        params = jsonargparse.Namespace(a=1, b=2, c=jsonargparse.Namespace(d=3, e=4))
        with mlflow.start_run() as run:
            mlflow.log_params(flatten_params(params))

        params.pop("c")
        run = mlflow.get_run(run_id=run.info.run_id)
        self.assertTrue(run_has_params(run, params))

        params.b = 3
        self.assertFalse(run_has_params(run, params))

    def test_stored_run_has_params_skip(self):
        params = jsonargparse.Namespace(a=1, b=2, c=jsonargparse.Namespace(d=3, e=4))
        with mlflow.start_run() as run:
            mlflow.log_params(flatten_params(params))

        params.b = 10
        params.c.d = 10
        run = mlflow.get_run(run_id=run.info.run_id)
        self.assertFalse(run_has_params(run, params))
        self.assertFalse(run_has_params(run, params, skip_keys=["c.d"]))
        self.assertFalse(run_has_params(run, params, skip_keys=["b"]))
        self.assertTrue(run_has_params(run, params, skip_keys=["b", "c.d"]))


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


class TestDeprecation(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.uri = os.path.abspath(os.path.join("sqlite:///", self.tempdir.name, "mlflow.db"))
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment("test_experiment")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_flatten_params_deprecation_warning(self):
        parser = jsonargparse.ArgumentParser()
        parser.add_argument("--a", default=1)
        args = parser.parse_args([])

        with self.assertWarns(DeprecationWarning) as cm:
            flatten_params(args)
        self.assertIn("run_registry", str(cm.warning))

    def test_run_has_params_deprecation_warning(self):
        parser = jsonargparse.ArgumentParser()
        parser.add_argument("--a", default=1)
        args = parser.parse_args([])

        with mlflow.start_run() as run:
            mlflow.log_param("a", 1)

        restored_run = mlflow.get_run(run_id=run.info.run_id)
        with self.assertWarns(DeprecationWarning) as cm:
            self.assertTrue(run_has_params(restored_run, args))
        self.assertIn("run_registry", str(cm.warning))
