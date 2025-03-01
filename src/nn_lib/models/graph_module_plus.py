import warnings
from typing import Iterable, List, Optional, Any, assert_never, Self

import pydot
from torch import nn
from torch.fx import symbolic_trace, GraphModule, Graph, Node

from nn_lib.models.graph_utils import prefix_all_nodes


class GraphModulePlus(GraphModule):
    """An extension of torch.fx.GraphModule that provides additional functionality."""

    ###############
    ## Factories ##
    ###############

    @staticmethod
    def new_from_copy(gm: GraphModule, name: Optional[str] = None) -> "GraphModulePlus":
        """Create a new GraphModulePlus from a copy of an existing GraphModule. The root module
        will share references to the attributes of the model, but the underlying graphs will be
        independent."""
        # Some special handling of graph-copying is required.
        new_graph = Graph()
        output_node = new_graph.graph_copy(gm.graph, {})
        if output_node is not None:
            new_graph.output(output_node)
        class_name = gm.__class__.__name__ if name is None else name
        return GraphModulePlus(root=gm, graph=new_graph, class_name=class_name)

    @staticmethod
    def new_from_trace(module: nn.Module) -> "GraphModulePlus":
        """Create a new GraphModulePlus by symbolic trace of a torch.nn.Module"""
        if isinstance(module, GraphModulePlus):
            return GraphModulePlus.new_from_copy(module)
        else:
            return GraphModulePlus.new_from_copy(symbolic_trace(module))

    def extract_subgraph(
        self, inputs: Optional[list[str | Node]] = None, output: Optional[str | Node] = None
    ) -> "GraphModulePlus":
        """Extract a subgraph by specifying the input and output nodes. Optionally leave the
        inputs and output unspecified to have them default to the inputs and output of the original
        model.

        Note: The returned GraphModule shares attributes/submodules/parameters with the original,
        but the graph is a new object. This allows for patterns like

            with frozen(module.extract_subgraph(...)):
                ...

        to freeze some subset of the original model."""
        if inputs is None:
            inputs = self.inputs
        if output is None:
            output = self.output_value
        new_module = GraphModulePlus.new_from_copy(self, name="Sub" + self.__class__.__name__)
        new_module.set_inputs_and_output(inputs, output)
        new_module.recompile()
        new_module.delete_all_unused_submodules()
        return new_module

    def replace_head(
        self,
        other: nn.Module | GraphModule,
        map_nodes: dict[str | Node, str | Node],
        this_prefix: Optional[str] = None,
        other_prefix: Optional[str] = None,
    ) -> "GraphModulePlus":
        """Create a new GraphModulePlus by merging the graph of this module with the graph of
        another Module. Keys in the map_nodes dict specify node names in this model, and values
        specify node names in the 'other' module which will be replaced by the nodes in this
        module. Inputs will come from 'self' and output will come from 'other'.

        To avoid naming collisions, set this_prefix and other_prefix to a string. The prefix also
        serves as an attribute name for each submodule. That is, if 'self' has attributes layer1
        and layer2 and 'this_prefix' is None, the new module will also have attributes layer1 and
        layer2. If 'this_prefix' is 'foo', the new module will have a submodule called 'foo' which
        will in turn contain attributes layer1 and layer2.
        """
        if not isinstance(other, GraphModule):
            other = GraphModulePlus.new_from_trace(other)

        # Since prefixes might be added below, we'll extract Node objects from their string names
        # first. Later, the prefix_all_nodes function will update these nodes' names in-place.
        map_keys = self._resolve_nodes(map_nodes.keys())
        map_values = other._resolve_nodes(map_nodes.values())
        map_nodes = dict(zip(map_keys, map_values))

        # Prepare container objects: we need an attribute dict and a graph to hold the new module.
        new_graph = Graph()
        new_attrs = nn.ModuleDict()
        new_node_lookup = {}

        def merge_with_prefix_helper(gm: GraphModule, prefix: Optional[str]):
            nonlocal new_graph, new_attrs, new_node_lookup
            if prefix is None:
                new_graph.graph_copy(gm.graph, val_map=new_node_lookup)
                duplicates = {key for key in gm._modules.keys() if key in new_attrs}
                if duplicates:
                    raise RuntimeError(f"Overwriting existing attributes: {duplicates}")
                new_attrs.update(gm._modules)
            else:
                new_graph.graph_copy(prefix_all_nodes(gm.graph, prefix), val_map=new_node_lookup)
                if prefix in new_attrs:
                    raise ValueError(f"Attribute name collision: {prefix}")
                new_attrs[prefix] = gm

        # Populate the new_graph and new_attrs objects with the nodes and submodules from 'self'
        # and from 'other'.
        merge_with_prefix_helper(self, this_prefix)
        merge_with_prefix_helper(other, other_prefix)

        # Perform the node substitutions
        for this_node, other_node in map_nodes.items():
            new_node_lookup[other_node].replace_all_uses_with(new_node_lookup[this_node])

        # TODO - this name will get long if we keep merging modules. Not really a problem, but
        #  would be nice to have multiple merges result in a more readable name.
        new_name = "Merged" + self.__class__.__name__ + other.__class__.__name__
        new_module = GraphModulePlus(root=new_attrs, graph=new_graph, class_name=new_name)

        new_module.set_inputs_and_output(
            inputs=[new_node_lookup[n] for n in self.inputs],
            output=new_node_lookup[other.output_value],
        )

        new_module.recompile()
        new_module.delete_all_unused_submodules()

        return new_module

    def squash_all_conv_batchnorm_pairs(self) -> "GraphModulePlus":
        """Squash all conv-batchnorm pairs in this model. Returns a new model with
        parameters/attributes shared with the original model *except* for the new conv layers.
        """
        new_module = GraphModulePlus.new_from_copy(
            self, name="Squashed" + self.__class__.__name__
        ).eval()

        # Find all conv-batchnorm pairs
        # TODO: handle functional calls like F.conv2d; currently we assume all convs and batchnorms
        #  are called as modules.
        conv_bn_pairs = []
        for node in new_module.graph.nodes:
            if node.op == "call_module" and isinstance(
                new_module.get_submodule(node.target), nn.Conv2d
            ):
                for user in node.users:
                    if user.op == "call_module" and isinstance(
                        new_module.get_submodule(user.target), nn.BatchNorm2d
                    ):
                        conv_bn_pairs.append((node, user))

        # Squash each pair
        for conv, bn in conv_bn_pairs:
            # Replace the conv node with a new conv node that has the bn parameters folded in
            common_prefix = []
            for conv_part, bn_part in zip(conv.target.split("."), bn.target.split(".")):
                if conv_part == bn_part:
                    common_prefix.append(conv_part)
                else:
                    break
            conv_unique_name = "_".join(conv.target.split(".")[len(common_prefix) :])
            bn_unique_name = "_".join(bn.target.split(".")[len(common_prefix) :])
            squashed_name = ".".join(common_prefix + [f"{conv_unique_name}_{bn_unique_name}"])
            conv_module = new_module.get_submodule(conv.target)
            bn_module = new_module.get_submodule(bn.target)
            squashed_conv = nn.utils.fuse_conv_bn_eval(conv_module, bn_module)
            new_module.add_submodule(squashed_name, squashed_conv)
            with new_module.graph.inserting_before(conv):
                new_node = new_module.graph.call_module(
                    squashed_name, args=conv.args, kwargs=conv.kwargs
                )
                bn.replace_all_uses_with(new_node)

        # Post-surgery, clean up the graph. This will remove all unused nodes, so any conv/bn
        # nodes that we squashed end up removed but only if they are no longer used in any other
        # part of the graph.
        new_module._clean_up_and_recompile()

        return new_module

    #############################
    ## Node management helpers ##
    #############################

    def _resolve_nodes(self, names: str | Node | Iterable[str | Node]) -> list[Node]:
        """Get all nodes in the graph with a given name. Raises a ValueError if any of the names
        are not present in the graph. Returns a list of nodes in the same order as the input names.
        """
        # Start by ensuring the type of 'names' is a list of strings
        if isinstance(names, str) or isinstance(names, Node):
            names = [names]
        names = [n.name if isinstance(n, Node) else n for n in names]

        # Keep track of which nodes are found
        lookup_node_by_name: dict[str, Optional[Node]] = {name: None for name in names}

        for node in self.graph.nodes:
            if node.name in lookup_node_by_name:
                lookup_node_by_name[node.name] = node

        missing_names = [name for name, node in lookup_node_by_name.items() if node is None]
        if missing_names:
            raise ValueError("Not all nodes are present in the graph:", missing_names)

        # This relies on the fact that python dicts are ordered, so the output will be in the same
        # order as the input.
        return list(lookup_node_by_name.values())

    @property
    def inputs(self) -> List[Node]:
        """Get all input nodes in the graph."""
        return [node for node in self.graph.nodes if node.op == "placeholder"]

    def _rm_output(self) -> Self:
        # Cast to a list so that we're not modifying the graph while iterating over it.
        for node in list(self.graph.nodes):
            if node.op == "output":
                self.graph.erase_node(node)
        return self

    @property
    def output(self) -> Node:
        """Get the output node of a graph (assumes there is exactly one). Note that this is just
        a pointer to the node whose value is output. Use output_value for the actual node whose
        value is returned."""
        return next(node for node in self.graph.nodes if node.op == "output")

    @property
    def output_value(self) -> Node:
        """Get the Node which is returned by the graph."""
        return self.output.args[0]

    def _update_all_inplace_ops(self, inplace: bool = False) -> Self:
        """Update any inplace operations (e.g. ReLU(inplace=True)) in a model. Set their
        'inplace' attribute to the given value. Setting inplace=False helps for instance by
        making the functions 'pure' and thus play nicer with torch.fx and torch.func.

        The module is itself modified in-place.

        Warning: this only works if the inplace ops are modules. That is, nn.ReLU will be
        updated, but function calls like F.relu will not be updated!
        """
        for node in self.graph.nodes:
            if node.op == "call_module":
                module = self.get_submodule(node.target)
                if hasattr(module, "inplace"):
                    module.inplace = inplace
        return self

    ########################
    ## Graph manipulation ##
    ########################

    def eliminate_dead(self, remove_unused_inputs: bool = True) -> Self:
        """Call self.graph.eliminate_dead_code() AND remove any lingering unused placeholder nodes,
        unless remove_unused_inputs=False.
        """
        # elminiate_dead_code() works backwards from outputs to inputs (it assumes the graph is
        # topologically sorted), removing any node that doesn't have users.
        self.graph.eliminate_dead_code()

        # For reasons unknown, eliminate_dead_code() doesn't remove unused placeholders. We'll do
        # that manually here.
        if remove_unused_inputs:
            # Cast to a list so that we're not modifying the graph while iterating over it.
            for node in list(self.graph.nodes):
                if node.op == "placeholder" and len(node.users) == 0:
                    self.graph.erase_node(node)

        return self

    def _clean_up_and_recompile(self):
        # The purpose of 'eliminate_dead' is to remove Nodes from the Graph that have no path to
        # any output node.
        self.eliminate_dead()
        # The purpose of 'recompile' is to take the modified Graph recompile it into a
        # self.forward() method. Without recompiling, changes to the graph don't actually affect
        # the module's function.
        self.recompile()
        # The purpose of 'delete_all_unused_submodules' is to remove attributes of self that point
        # to other nn.Modules which are no longer used after the graph has been modified.
        self.delete_all_unused_submodules()

    def set_inputs(self, inputs: list[str | Node]) -> Self:
        """Set the inputs of this graph by finding nodes of the given name(s) and replacing them
        with placeholders. Modifies the graph attribute in-place."""
        # For each named input, erase any existing nodes of the same name and replace them with a
        # new placeholder node.
        dont_remove = []
        for node in self._resolve_nodes(inputs):
            if node.op == "placeholder":
                dont_remove.append(node)
                continue
            else:
                with self.graph.inserting_before(node):
                    new_placeholder = self.graph.placeholder(node.name, node.type)
                    node.replace_all_uses_with(new_placeholder)
                    self.graph.erase_node(node)
                    # Handle potential name collision by reinforcing the new node name now that
                    # the old node is deleted.
                    if new_placeholder.name != node.name:
                        new_placeholder.name = node.name
                    dont_remove.append(new_placeholder)

        self._clean_up_and_recompile()

        return self

    def set_output(self, output: str | Node) -> Self:
        """Remove all preexisting outputs and set the output of a graph to the node of the given
        name."""
        # Find the named node to be the arg to a new output node
        node_to_output = self._resolve_nodes(output)[0]

        # Remove all preexisting outputs.
        self._rm_output()

        with self.graph.inserting_after():
            self.graph.output(node_to_output)

        self._clean_up_and_recompile()

        return self

    def set_dict_outputs(self, outputs: Iterable[str | Node]) -> Self:
        """Modify the Graph by adding a new node which collects multiple outputs in a dict. This
        new node will then become the output of the graph.
        """
        # Find the named nodes to be the args to a new output node
        nodes_to_output = self._resolve_nodes(outputs)
        names = [node.name for node in nodes_to_output]

        # Remove all preexisting outputs.
        self._rm_output()

        with self.graph.inserting_after():
            # Create a new node which collects the outputs into a dict
            collector_node = self.graph.call_function(
                the_function=dict,
                kwargs={name: node for name, node in zip(names, nodes_to_output)},
            )
            # Set the new 'collector' node as the output of the graph
            self.graph.output(collector_node)

        self._clean_up_and_recompile()

        return self

    def set_inputs_and_output(self, inputs: list[str | Node], output: str | Node) -> Self:
        """Set both the inputs and output of this graph. Modifies the graph attribute in-place."""
        if output in inputs:
            raise ValueError("Output node cannot also be an input node.")
        self.set_output(output)
        self.set_inputs(inputs)

        return self

    ############################################
    ## Debugging and visualization utilities ##
    ############################################

    def to_dot(self) -> pydot.Dot:
        dot = pydot.Dot()
        for node in self.graph.nodes:
            dot.add_node(pydot.Node(node.name))
            for user in node.users:
                dot.add_edge(pydot.Edge(node.name, user.name))
        return dot

    def step_through_call(self, context: Optional[dict] = None, callback=None) -> Any:
        """Line-by-line debuggable model evaluation allowing the graph itself to be debugged.

        The idea is that

             output = model(**context)

        has the same behavior as

            output = model.step_through_call(context)

        but the latter allows for stepping through the graph node-by-node, inspecting the state of
        the model at each step, and callbacks.
        """

        def _get_arg(arg: Any):
            if isinstance(arg, Node):
                return context[arg.name]
            elif isinstance(arg, list):
                return [_get_arg(a) for a in arg]
            elif isinstance(arg, dict):
                return {k: _get_arg(v) for k, v in arg.items()}
            else:
                return arg

        for node in self.graph.nodes:
            match node.op:
                case "placeholder":
                    assert node.name in context, f"Missing input {node.name}"
                case "get_attr":
                    obj = self
                    for part in node.target.split("."):
                        obj = getattr(obj, part)
                    context[node.name] = obj
                case "call_module":
                    module = self.get_submodule(node.target)
                    args = [_get_arg(arg) for arg in node.args]
                    kwargs = {k: _get_arg(v) for k, v in node.kwargs.items()}
                    context[node.name] = module(*args, **kwargs)
                case "call_method":
                    self_obj, *args = [_get_arg(arg) for arg in node.args]
                    kwargs = {k: _get_arg(v) for k, v in node.kwargs.items()}
                    method = getattr(self_obj, node.target)
                    context[node.name] = method(*args, **kwargs)
                case "call_function":
                    the_function = node.target
                    args = [_get_arg(arg) for arg in node.args]
                    kwargs = {k: _get_arg(v) for k, v in node.kwargs.items()}
                    context[node.name] = the_function(*args, **kwargs)
                case "output":
                    args = [_get_arg(arg) for arg in node.args]
                    return args[0]
                case _:
                    assert_never(node.op)

            if callback is not None:
                callback(node, context[node.name])
