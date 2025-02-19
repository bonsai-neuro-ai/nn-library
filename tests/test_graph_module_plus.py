import unittest
from warnings import catch_warnings

import torch
from torchvision.models.resnet import resnet18, resnet34
from nn_lib.models.graph_module_plus import GraphModulePlus
from nn_lib.models.utils import frozen


class TestGraphModulePlus(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize a GraphModulePlus instance and any necessary nodes
        self.reference_module = resnet18()
        self.gm = GraphModulePlus.new_from_trace(self.reference_module)
        self.dummy_input = torch.randn(1, 3, 224, 224)

    def test_inputs(self):
        inputs = self.gm.inputs
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "x")

    def test_output(self):
        self.assertEqual(self.gm.output.name, "output")
        self.assertEqual(self.gm.output_value.name, "fc")

    def test_set_inputs_no_eliminate_dead(self):
        # Without eliminate_dead, the method should raise a UserWarning when it tries to get rid
        # of the previous input "x"
        with catch_warnings(record=True) as w:
            self.gm.set_inputs(["maxpool"], eliminate_dead=False)
        self.assertEqual(len(w), 1)
        self.assertIn("Could not remove input node x.", str(w[0].message))

    def test_set_inputs_eliminate_dead(self):
        # With eliminate_dead, the method should just work without warnings
        with catch_warnings(record=True) as w:
            self.gm.set_inputs(["maxpool"], eliminate_dead=True)
        self.assertEqual(len(w), 0)
        inputs = self.gm.inputs
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "maxpool")

    def test_set_output(self):
        num_nodes_before = len(list(self.gm.graph.nodes))
        the_node = self.gm._resolve_nodes("layer1_1_relu")[0]
        self.gm.set_output("layer1_1_relu")
        self.assertEqual(self.gm.output_value, the_node)

        num_nodes_after = len(list(self.gm.graph.nodes))
        self.assertLess(num_nodes_after, num_nodes_before)

    def test_set_dict_outputs(self):
        self.gm.set_dict_outputs(outputs=["add_1", "add_2", "add_3"])
        out = self.gm(self.dummy_input)
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(len(out), 3)
        self.assertIn("add_1", out)
        self.assertIn("add_2", out)
        self.assertIn("add_3", out)
        self.assertTrue(isinstance(out["add_1"], torch.Tensor))
        self.assertTrue(isinstance(out["add_2"], torch.Tensor))
        self.assertTrue(isinstance(out["add_3"], torch.Tensor))

    def test_set_inputs_and_output_noop(self):
        num_nodes_before = len(list(self.gm.graph.nodes))
        dummy_output_before = self.gm(self.dummy_input)

        self.gm.set_inputs_and_output(self.gm.inputs, self.gm.output_value)

        num_nodes_after = len(list(self.gm.graph.nodes))
        dummy_output_after = self.gm(self.dummy_input)

        self.assertEqual(num_nodes_before, num_nodes_after)
        self.assertTrue(torch.allclose(dummy_output_before, dummy_output_after))

    def test_squash_conv_bn(self):
        self.gm.eval()
        output_before = self.gm(self.dummy_input)
        new_gm = self.gm.squash_all_conv_batchnorm_pairs()
        output_after = new_gm(self.dummy_input)

        def is_softmax_close(a, b):
            return torch.allclose(torch.softmax(a, dim=-1), torch.softmax(b, dim=-1), atol=1e-6)

        self.assertTrue(is_softmax_close(output_before, output_after))

        # The thing about batchnorm is that in 'training' mode, it updates itself on any forward
        # call. So we can check it all BN were eliminated by verifying that outputs are not changing
        # during train mode.
        new_gm.train()
        new_gm(self.dummy_input)
        output_after_2 = new_gm(self.dummy_input)
        self.assertTrue(is_softmax_close(output_after, output_after_2))

    def test_freeze_subgraph(self):
        params_before = {k: v.clone() for k, v in self.gm.named_parameters()}
        with frozen(self.gm.extract_subgraph(inputs=["add_1"], output="add_5")):
            opt = torch.optim.SGD(self.gm.parameters(), lr=0.1)
            for _ in range(10):
                opt.zero_grad()
                self.gm(self.dummy_input).sum().backward()
                opt.step()

        params_after = {k: v.clone() for k, v in self.gm.named_parameters()}

        for k in params_before:
            # We chose 'add_1' and 'add_5' above because they bracket the layer2 and layer3 parts
            # of the resnet18 model. So we expect the parameters in those blocks to be frozen.
            if k.startswith("layer2") or k.startswith("layer3"):
                self.assertTrue(
                    torch.allclose(params_before[k], params_after[k]),
                    f"Expected {k} to be frozen",
                )
            else:
                self.assertFalse(
                    torch.allclose(params_before[k], params_after[k]),
                    f"Expected {k} to be updated",
                )

        # Outside the with context, train again and assert that everything changed
        opt = torch.optim.SGD(self.gm.parameters(), lr=0.1)
        for _ in range(10):
            opt.zero_grad()
            self.gm(self.dummy_input).sum().backward()
            opt.step()

        params_after_after = {k: v.clone() for k, v in self.gm.named_parameters()}

        for k in params_after:
            self.assertFalse(
                torch.allclose(params_after[k], params_after_after[k]),
                f"Expected {k} to be updated",
            )

    def test_replace_head_name_collision(self):
        model2 = GraphModulePlus.new_from_trace(resnet34())

        # Currently, we actually expect name collisions to be triggered if model1 and model2 have
        # overlapping attribute names, even if sub-models model1[input:add_1] and
        # model2[add_2:output] do not. This could be fixed someday, but for now this test is
        # asserting that the current overly-cautious behavior is correct.
        with self.assertRaises(RuntimeError):
            stitched_model = self.gm.replace_head(model2, {"add_1": "add_2"})

    def test_replace_head_prefix(self):
        # Repeat the test test_replace_head_name_collision, but prefix the model attributes to avoid
        # name collisions
        model2 = GraphModulePlus.new_from_trace(resnet34())

        with catch_warnings(record=True) as w:
            stitched_model = self.gm.replace_head(
                model2, {"add_1": "add_2"}, this_prefix="model1", other_prefix="model2"
            )

        self.assertEqual(len(w), 0, "Expected no warnings but got: " + str(w))

        # Check that the model runs
        out = stitched_model(self.dummy_input)

        # Check that it has the expected input and output names
        self.assertEqual(stitched_model.inputs[0].name, "model1_x")
        self.assertEqual(stitched_model.output_value.name, "model2_fc")


if __name__ == "__main__":
    unittest.main()
