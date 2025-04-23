import os
import unittest
from tempfile import TemporaryDirectory

import mlflow

from nn_lib.utils.mlflow import save_as_artifact, load_artifact


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
