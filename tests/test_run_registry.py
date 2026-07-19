"""Tests for run_registry: canonicalization, hashing, and MLflow-backed dedup.

Integration tests use a real MLflow file-based tracking store in a tmp dir -- no mocking of
mlflow -- so they exercise the actual param stringification, truncation, tags, statuses,
and search behavior of the installed mlflow.
"""

import dataclasses
import enum
import functools
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import yaml
from jsonargparse import ArgumentParser, Namespace, lazy_instance

from nn_lib.utils.run_registry import (
    PARAMS_HASH_MLFLOW_TAG,
    RunIndex,
    canonical_strings,
    dump_config_yaml,
    flatten,
    hash_spec,
    logged_run,
    to_plain,
    select_spec,
)

####################
# Shared fixtures  #
####################


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclasses.dataclass
class Metric:
    centered: bool = True
    p: float = 2.0


class NiceRepr:
    def __str__(self):
        return "NiceRepr(seed=1)"


class DefaultRepr:
    pass


def _plain_python_function():
    pass


def _specs():
    # NB: metric is a plain dict, not Metric(...): instantiated dataclasses are rejected by
    # to_plain because their fields alone cannot carry class identity (see
    # TestCanonicalizeToPlain.test_dataclass_instances_raise).
    base = dict(model="resnet18", metric={"centered": True, "p": 2.0}, mode=Mode.FAST)
    return [{**base, "layer": f"conv{i}"} for i in range(3)]


def make_parser():
    def f(root: Path, metric: Metric = Metric(), name: str = "resnet", tags: dict = {}):
        pass

    parser = ArgumentParser(exit_on_error=False)
    parser.add_function_arguments(f)
    return parser


class Backbone:
    """Base class parsed in jsonargparse's subclass mode ({class_path, init_args})."""

    def __init__(self, hidden: int = 8):
        self.hidden = hidden


class ConvBackbone(Backbone):
    pass


class AttnBackbone(Backbone):
    pass


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def make_subclass_parser(with_config: bool = False) -> ArgumentParser:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(Backbone, "model")
    if with_config:
        parser.add_argument("--config", action="config")
    return parser


####################
# Canonicalization #
####################


class TestCanonicalizeToPlain(unittest.TestCase):
    def test_primitives_pass_through(self):
        result = to_plain({"a": 1, "b": 2.5, "c": "x", "d": True, "e": None})
        self.assertEqual(
            result,
            {
                "a": 1,
                "b": 2.5,
                "c": "x",
                "d": True,
                "e": None,
            },
        )

    def test_path_enum_set(self):
        plain = to_plain({"p": Path("/data"), "m": Mode.FAST, "s": {3, 1, 2}})
        self.assertEqual(plain, {"p": "/data", "m": "FAST", "s": [1, 2, 3]})

    def test_nested_dicts_lists_and_namespaces(self):
        ns = Namespace(
            a=1, sub=Namespace(x=2, y=Namespace(z=3)), d={"k": {"deep": 5}}, l=[1, (2, 3)]
        )
        plain = to_plain(ns)
        self.assertEqual(
            plain,
            {
                "a": 1,
                "sub": {"x": 2, "y": {"z": 3}},
                "d": {"k": {"deep": 5}},
                "l": [1, [2, 3]],
            },
        )

    def test_dataclass_instances_raise(self):
        """Instantiated dataclasses are rejected. Serializing fields alone would drop class
        identity, so two subclass-mode dataclasses with identical fields would silently
        hash-collide (and never match their pre-instantiation {class_path, init_args} form
        either). Fail loudly instead; spec authors should use the pre-instantiation config
        or an equivalent plain dict."""
        with self.assertRaises(TypeError) as cm:
            to_plain({"metric": Metric(p=3.5)})
        self.assertIn("dataclass", str(cm.exception))
        self.assertIn("'metric'", str(cm.exception))  # error names the offending key

    def test_deterministic_str_object_accepted(self):
        result = to_plain({"x": NiceRepr()})
        self.assertEqual(result, {"x": "NiceRepr(seed=1)"})

    def test_address_bearing_objects_raise(self):
        bad_objects = [
            DefaultRepr(),
            lambda x: x,
            functools.partial(_plain_python_function),
            object(),
        ]
        for bad in bad_objects:
            with self.assertRaises(TypeError) as cm:
                to_plain({"bad": bad})
            self.assertIn("memory address", str(cm.exception))
            self.assertIn("'bad'", str(cm.exception))  # error names the offending key

    def test_yaml_roundtrip(self):
        spec = {"a": 1, "root": Path("/data"), "mode": Mode.SLOW, "nested": {"b": [1, 2]}}
        self.assertDictEqual(yaml.safe_load(dump_config_yaml(spec)), to_plain(spec))


class TestFlatteningAndNamespaceConvention(unittest.TestCase):
    def test_flatten_matches_namespace_items_for_pure_namespace_configs(self):
        """For parser-generated configs (all nesting is Namespace), our flat keys
        must equal jsonargparse's own `Namespace.items()` dotted-key convention."""
        cfg = make_parser().parse_args(["--root", "/data", "--metric.p", "3.5"])
        ours = set(flatten(to_plain(cfg)).keys())
        theirs = set(dict(cfg.items()).keys())
        # `tags` is a dict leaf: items() keeps it opaque; empty dict flattens away.
        theirs.discard("tags")
        self.assertEqual(ours, theirs)


class TestHashingAndCanonicalStrings(unittest.TestCase):
    def test_cli_config_and_program_dict_hash_identically(self):
        """The same run described via CLI parsing or via a hand-built dict must
        produce the same canonical strings and hash."""
        cfg = make_parser().parse_args(["--root", "/data", "--metric.p", "3.5"])
        program = {"root": Path("/data"), "metric": {"centered": True, "p": 3.5}, "name": "resnet"}
        self.assertEqual(canonical_strings(cfg), canonical_strings(program))
        self.assertEqual(hash_spec(cfg), hash_spec(program))

    def test_hash_key_subset_and_missing_keys(self):
        spec = {"a": 1, "b": 2}
        self.assertEqual(hash_spec(spec, keys=["a"]), hash_spec({"a": 1, "b": 999}, keys=["a"]))
        with self.assertRaises(KeyError):
            hash_spec({"a": 1}, keys=["a", "zzz"])

    def test_hash_is_order_insensitive_but_value_sensitive(self):
        self.assertEqual(hash_spec({"a": 1, "b": 2}), hash_spec({"b": 2, "a": 1}))
        self.assertNotEqual(hash_spec({"a": 1}), hash_spec({"a": 2}))
        # int 1 vs str "1": both stringify to "1" -- documented behavior, pin it.
        self.assertEqual(hash_spec({"a": 1}), hash_spec({"a": "1"}))


#############################################
# jsonargparse class_path/init_args pattern #
#############################################


class TestClassPathInitArgsPattern(unittest.TestCase):
    """jsonargparse subclass-mode configs.

    The supported spec form is the *pre-instantiation* config, where a subclass-typed
    argument parses to Namespace(class_path=..., init_args=Namespace(...)). These tests pin:
    canonical structure, class_path participating in run identity, CLI/program-built parity,
    the config.yaml round trip, and loud failure for post-instantiation objects.
    """

    def test_pre_instantiation_subclass_config_canonicalizes(self):
        cfg = make_subclass_parser().parse_args(
            [f"--model={_class_path(ConvBackbone)}", "--model.init_args.hidden=16"]
        )
        self.assertEqual(
            to_plain(cfg),
            {"model": {"class_path": _class_path(ConvBackbone), "init_args": {"hidden": 16}}},
        )
        self.assertEqual(
            canonical_strings(cfg),
            {
                "model.class_path": _class_path(ConvBackbone),
                "model.init_args.hidden": "16",
            },
        )

    def test_class_path_is_part_of_run_identity(self):
        parser = make_subclass_parser()
        spec_a = parser.parse_args([f"--model={_class_path(ConvBackbone)}"])
        spec_b = parser.parse_args([f"--model={_class_path(AttnBackbone)}"])
        # Identical __init__ signatures -> identical flat key sets, so a grid over these
        # subclasses satisfies the RunIndex shared-key-set contract...
        self.assertEqual(canonical_strings(spec_a).keys(), canonical_strings(spec_b).keys())
        # ...but identical init_args must NOT dedup across different class_path.
        self.assertNotEqual(hash_spec(spec_a), hash_spec(spec_b))

    def test_cli_and_program_built_subclass_specs_hash_identically(self):
        cli = make_subclass_parser().parse_args(
            [f"--model={_class_path(ConvBackbone)}", "--model.init_args.hidden=16"]
        )
        program = {"model": {"class_path": _class_path(ConvBackbone), "init_args": {"hidden": 16}}}
        self.assertEqual(canonical_strings(cli), canonical_strings(program))
        self.assertEqual(hash_spec(cli), hash_spec(program))

    def test_subclass_config_yaml_roundtrip(self):
        """dump_config_yaml of a subclass spec must reload via --config to the same hash."""
        parser = make_subclass_parser(with_config=True)
        spec1 = select_spec(
            parser.parse_args(
                [f"--model={_class_path(ConvBackbone)}", "--model.init_args.hidden=16"]
            )
        )
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(dump_config_yaml(spec1))
            spec2 = select_spec(parser.parse_args([f"--config={config_path}"]))
        self.assertEqual(canonical_strings(spec1), canonical_strings(spec2))
        self.assertEqual(hash_spec(spec1), hash_spec(spec2))

    def test_instantiated_objects_fail_loudly(self):
        parser = make_subclass_parser()
        cfg = parser.parse_args([f"--model={_class_path(ConvBackbone)}"])
        init = parser.instantiate_classes(cfg)
        self.assertIsInstance(init.model, ConvBackbone)  # sanity: post-instantiation form
        with self.assertRaises(TypeError) as cm:
            to_plain(init)
        # The error must name the offending key and steer users back to the
        # pre-instantiation config.
        self.assertIn("'model'", str(cm.exception))
        self.assertIn("instantiate_classes", str(cm.exception))

    def test_live_instance_default_fails_loudly_with_guidance(self):
        """A live instance used as an argument default is kept as-is in the parsed config when
        the argument isn't overridden -- surprising, because the user never called
        instantiate_classes. The error must name the offending key and point at the intended
        fixes (lazy_instance, or a class_path/init_args dict default). NB: pinned against the
        installed jsonargparse's instance-default behavior; if this fails at the isinstance
        sanity check, the installed version serializes instance defaults itself and this guard is
        moot."""
        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=Backbone, default=lazy_instance(ConvBackbone, hidden=4))
        cfg = parser.parse_args([])
        instantiated_cfg = parser.instantiate(cfg)
        self.assertIsInstance(instantiated_cfg.model, ConvBackbone)  # jsonargparse instantiated the default

        # Trying to serialize the instantiated cfg should raise errors
        with self.assertRaises(TypeError) as cm:
            to_plain(instantiated_cfg)
        msg = str(cm.exception)
        self.assertIn("'model'", msg)
        self.assertIn("lazy_instance", msg)

        # ..but instantiating the raw/lazy config is fine:
        _ = to_plain(cfg)


##########################
# MLflow integration     #
##########################


class TestMLFlowIntegration(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.uri = f"sqlite:///{self.tempdir.name}/mlflow.db"
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment("test-exp")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_logged_run_then_index_dedups_via_tag(self):
        specs = _specs()
        with logged_run(specs[0]):
            mlflow.log_metric("distance", 0.5)

        index = RunIndex.from_experiment(keys=specs[0])
        self.assertIn(specs[0], index)
        self.assertNotIn(specs[1], index)
        self.assertEqual(list(index.pending(specs)), specs[1:])

        # the tag really is the primary path
        run = mlflow.get_run(index.lookup(specs[0]))
        self.assertEqual(run.data.tags[PARAMS_HASH_MLFLOW_TAG], hash_spec(specs[0]))

    def test_legacy_run_without_tag_dedups_via_param_reconstruction(self):
        spec = _specs()[0]
        with mlflow.start_run():  # simulate a run logged before run_registry existed
            mlflow.log_params(flatten(to_plain(spec)))
        index = RunIndex.from_experiment(keys=spec)
        self.assertIn(spec, index)

    def test_truncated_param_still_dedups_via_tag(self):
        spec = {"model": "resnet18", "notes": "x" * 7000}  # > mlflow's 6000-char limit
        with logged_run(spec):
            pass
        run = mlflow.get_run(mlflow.search_runs(output_format="list")[0].info.run_id)
        self.assertLess(len(run.data.params["notes"]), 7000)  # mlflow truncated it...
        index = RunIndex.from_experiment(keys=spec)
        self.assertIn(spec, index)  # ...but the tag dedups anyway

    def test_failed_runs_are_not_duplicates_by_default(self):
        spec = _specs()[0]
        with self.assertRaises(RuntimeError):
            with logged_run(spec):
                raise RuntimeError("boom")

        index = RunIndex.from_experiment(keys=spec)
        self.assertNotIn(spec, index)  # FAILED -> retry
        index_all = RunIndex.from_experiment(keys=spec, statuses=("FINISHED", "RUNNING", "FAILED"))
        self.assertIn(spec, index_all)

    def test_failed_run_logs_error_artifact_and_config_yaml(self):
        spec = _specs()[0]
        with self.assertRaises(ValueError):
            with logged_run(spec):
                raise ValueError("kaboom")
        run = mlflow.search_runs(output_format="list")[0]
        client = mlflow.MlflowClient()
        names = {a.path for a in client.list_artifacts(run.info.run_id)}
        self.assertTrue({"config.yaml", "error.log"} <= names)

        with TemporaryDirectory() as tmp_path:
            local = client.download_artifacts(run.info.run_id, "config.yaml", str(tmp_path))
            self.assertEqual(yaml.safe_load(open(local)), to_plain(spec))
            local = client.download_artifacts(run.info.run_id, "error.log", str(tmp_path))
            self.assertIn("kaboom", open(local).read())

    def test_running_runs_count_as_duplicates(self):
        spec = _specs()[0]
        run = mlflow.start_run()  # leave it RUNNING
        try:
            mlflow.log_params(flatten(to_plain(spec)))
            mlflow.set_tag(PARAMS_HASH_MLFLOW_TAG, hash_spec(spec))
        finally:
            mlflow.end_run()  # end for cleanliness of the fluent API state
        # mark it RUNNING again at the store level to simulate a concurrent worker
        mlflow.MlflowClient().set_terminated(run.info.run_id, status="RUNNING")
        index = RunIndex.from_experiment(keys=spec)
        self.assertIn(spec, index)

    def test_index_add_guards_same_process_duplicates(self):
        specs = _specs()
        index = RunIndex.from_experiment(keys=specs[0])
        for spec in index.pending(specs):
            with logged_run(spec, index=index):
                pass
        # nothing pending afterwards, even without re-querying mlflow
        self.assertEqual(list(index.pending(specs)), [])
        self.assertEqual(len(index), len(specs))

    def test_logged_run_rejects_spec_keys_outside_index(self):
        specs = _specs()
        index = RunIndex.from_experiment(keys=specs[0])
        bad = {**specs[0], "batch_size": 128}  # execution param snuck into the spec
        with self.assertRaises(KeyError) as cm:
            with logged_run(bad, index=index):
                pass
        self.assertIn("batch_size", str(cm.exception))

    def test_end_to_end_cli_parsed_grid(self):
        """Full loop: parse a config from CLI args, derive per-run specs, run the
        grid twice; second pass finds nothing to do."""
        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=str)
        parser.add_argument("--metric", type=Metric, default=Metric())
        args = parser.parse_args(["--model", "vit_b_16", "--metric.p", "1.0"])

        def make_specs():
            base = args.as_dict()
            return [{**base, "layer": layer} for layer in ("conv1", "conv2")]

        for expected_todo in (2, 0):
            specs = make_specs()
            index = RunIndex.from_experiment(keys=specs[0])
            todo = list(index.pending(specs))
            self.assertEqual(len(todo), expected_todo)
            for spec in todo:
                with logged_run(spec, index=index):
                    mlflow.log_metric("distance", 1.0)

    def test_cli_config_yaml_roundtrip_via_logged_run_artifact(self):
        """Round-trip: CLI args -> logged config.yaml -> --config=config.yaml.

        Verifies that parser loading from dumped yaml reproduces the same run spec
        (canonical params) and deterministic metric.
        """
        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=str)
        parser.add_argument("--root", type=Path)
        parser.add_argument("--mode", type=Mode)
        parser.add_argument("--metric", type=Metric, default=Metric())
        parser.add_argument("--layers", type=list[int])
        parser.add_argument("--options", type=dict, default={})
        parser.add_argument("--enabled", type=bool, default=True)
        parser.add_argument("--config", action="config")

        cli_args = [
            "--model",
            "vit_b_16",
            "--root",
            "/data/imagenet",
            "--mode",
            "SLOW",
            "--metric.p",
            "1.5",
            "--metric.centered",
            "false",
            "--layers",
            "[1, 2, 4]",
            "--options",
            '{"alpha": 0.1, "nested": {"k": "v"}}',
            "--enabled",
            "true",
        ]

        # First run from direct CLI arguments.
        spec1 = select_spec(parser.parse_args(cli_args))
        with logged_run(spec1):
            params1 = canonical_strings(spec1)
            # Deterministic metric computed from canonicalized params.
            metric1 = float(len(params1) + sum(len(v) for v in params1.values()))
            mlflow.log_metric("roundtrip_score", metric1)

        run1 = mlflow.search_runs(output_format="list")[0]
        client = mlflow.MlflowClient()

        with TemporaryDirectory() as tmp_path:
            config_path = client.download_artifacts(run1.info.run_id, "config.yaml", str(tmp_path))

            # Second run by reloading via --config=config.yaml.
            spec2 = select_spec(parser.parse_args([f"--config={config_path}"]))
            with logged_run(spec2):
                params2 = canonical_strings(spec2)
                metric2 = float(len(params2) + sum(len(v) for v in params2.values()))
                mlflow.log_metric("roundtrip_score", metric2)

        runs = mlflow.search_runs(output_format="list")
        self.assertEqual(len(runs), 2)

        # Identify first and second runs robustly (newest first from search_runs).
        by_id = {r.info.run_id: r for r in runs}
        run1 = by_id[run1.info.run_id]
        run2 = next(r for r in runs if r.info.run_id != run1.info.run_id)

        # Same canonical params => same hash tag and same MLflow param strings.
        self.assertEqual(canonical_strings(spec1), canonical_strings(spec2))
        self.assertEqual(
            run1.data.tags[PARAMS_HASH_MLFLOW_TAG],
            run2.data.tags[PARAMS_HASH_MLFLOW_TAG],
        )
        self.assertEqual(run1.data.params, run2.data.params)

        # Metric parity confirms behavior parity for this deterministic computation.
        self.assertAlmostEqual(
            run1.data.metrics["roundtrip_score"],
            run2.data.metrics["roundtrip_score"],
        )

    def test_subclass_grid_dedup_uses_class_path_identity(self):
        """Grid over two subclasses with identical init_args: finishing one run must not
        mark the other as done, in the live index or a fresh one built from the store."""
        parser = make_subclass_parser()
        conv, attn = (
            parser.parse_args([f"--model={_class_path(cls)}", "--model.init_args.hidden=8"])
            for cls in (ConvBackbone, AttnBackbone)
        )
        index = RunIndex.from_experiment(keys=conv)
        with logged_run(conv, index=index):
            pass
        self.assertIn(conv, index)
        self.assertNotIn(attn, index)
        # A fresh index built from the MLflow store agrees (tag-based path).
        fresh = RunIndex.from_experiment(keys=conv)
        self.assertIn(conv, fresh)
        self.assertNotIn(attn, fresh)


class TestSelectSpec(unittest.TestCase):
    def test_default_drops_config_key_from_namespace(self):
        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=str)
        parser.add_argument("--config", action="config")
        args = parser.parse_args(["--model", "resnet18"])

        selected = select_spec(args)
        self.assertNotIn("config", selected)
        self.assertEqual(selected["model"], "resnet18")

    def test_default_drops_config_key_from_mapping(self):
        selected = select_spec({"model": "resnet18", "config": "/tmp/run.yaml"})
        self.assertEqual(selected, {"model": "resnet18"})

    def test_explicit_drop_removes_runtime_keys(self):
        raw = {
            "model": "vit_b_16",
            "metric": {"p": 1.0, "centered": True},
            "device": "cuda:0",
            "num_workers": 8,
            "seed": 123,
        }
        selected = select_spec(raw, drop=["device", "num_workers"])
        self.assertEqual(
            selected,
            {
                "model": "vit_b_16",
                "metric": {"p": 1.0, "centered": True},
                "seed": 123,
            },
        )

    def test_keep_overrides_default_drop_for_config(self):
        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=str)
        parser.add_argument("--config", action="config")
        args = parser.parse_args(["--model", "resnet18"])

        # Override the default; use drop=None to tell it *not* to drop the config
        selected = select_spec(args, drop=None)
        self.assertIn("config", selected)

    def test_output_is_compatible_with_registry_canonicalization(self):
        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--root", type=Path)
        parser.add_argument("--mode", type=Mode)
        parser.add_argument("--metric", type=Metric, default=Metric())
        parser.add_argument("--config", action="config")
        args = parser.parse_args(["--root", "/data", "--mode", "SLOW", "--metric.p", "3.5"])

        spec = select_spec(args)
        # No crash and stable canonical output for downstream hashing/logging.
        strings = canonical_strings(spec)
        self.assertIn("root", strings)
        self.assertIn("mode", strings)
        self.assertIn("metric.p", strings)
        self.assertIn("metric.centered", strings)
        self.assertNotIn("config", strings)

    def test_drop_missing_key_is_noop(self):
        selected = select_spec({"model": "resnet18"}, drop=["does_not_exist"])
        self.assertEqual(selected, {"model": "resnet18"})

    def test_dropping_nested_key_raises_error(self):
        params = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        spec = select_spec(params, drop=["b"])
        self.assertDictEqual(spec, {"a": 1, "c": {"d": 3, "e": 4}})

        with self.assertRaises(NotImplementedError):
            _ = select_spec(params, drop=["c.d"])


if __name__ == "__main__":
    unittest.main()
