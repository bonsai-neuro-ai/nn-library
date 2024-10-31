import pydot
import warnings
import torch
from torch import nn
from torch.fx import GraphModule, Graph, Node, symbolic_trace
from typing import Iterable, Optional, Any, assert_never
from nn_lib.models.utils import squash_conv_batchnorm
from copy import deepcopy


__all__ = [
    "GraphModule",
    "Graph",
    "Node",
    "get_nodes_by_name",
    "get_subgraph",
    "prefix_all_nodes",
    "set_dict_outputs_by_name",
    "set_inputs_and_output_by_name",
    "step_through_call",
    "stitch_graphs",
    "squash_all_conv_batchnorm_pairs",
    "symbolic_trace",
    "to_dot",
]


def get_nodes_by_name(graph: Graph, names: str | Iterable[str]) -> list[Node]:
    """Get nodes from a graph by name. The name argument may be a string or an iterable of strings.

    Raises a ValueError after iteration is complete if not all requested names were present in
    the graph.
    """
    if isinstance(names, str):
        names = [names]
    names = list(names)
    lookup_node_by_name: dict[str, Optional[Node]] = {name: None for name in names}

    for node in graph.nodes:
        if node.name in lookup_node_by_name:
            lookup_node_by_name[node.name] = node

    missing_names = [name for name, node in lookup_node_by_name.items() if node is None]
    if missing_names:
        raise ValueError("Not all nodes are present in the graph:", missing_names)

    return list(lookup_node_by_name.values())


def _copy_module_new_graph(graph_module: GraphModule, name: Optional[str] = None) -> GraphModule:
    """Get a new GraphModule which shares attribute/submodule references with the original, but has
    a separate graph object. This is useful for making modifications to the graph without affecting
    the original module.
    """
    new_graph = Graph()
    output_node = new_graph.graph_copy(graph_module.graph, {})
    if output_node is not None:
        new_graph.output(output_node)
    class_name = graph_module.__class__.__name__ if name is None else name
    return GraphModule(root=graph_module, graph=new_graph, class_name=class_name)


def _set_inputs_by_name(graph: Graph, inputs: Iterable[str]) -> None:
    """Set the inputs of a graph by finding nodes of the given name(s) and replacing them with
    placeholders."""
    # For each named input, erase any existing nodes of the same name and replace them with a
    # new placeholder node.
    for node in get_nodes_by_name(graph, inputs):
        if node.op == "placeholder":
            continue
        else:
            with graph.inserting_before(node):
                new_placeholder = graph.placeholder(node.name, node.type)
                node.replace_all_uses_with(new_placeholder)
                graph.erase_node(node)

    # Remove all other preexisting inputs.
    for node in list(graph.nodes):
        if node.op == "placeholder" and node.name not in inputs:
            try:
                graph.erase_node(node)
            except RuntimeError:
                warnings.warn(
                    f"Could not remove input node {node.name}. Either this node is still genuinely "
                    f"in use, or you should call eliminate_dead_code() before set_inputs_by_name()."
                )


def _set_output_by_name(graph: Graph, output: str) -> None:
    """Remove all preexisting outputs and set the output of a graph to the node of the given name."""
    # Find the named node to be the arg to a new output node
    node_to_output = get_nodes_by_name(graph, output)[0]

    # Remove all preexisting outputs
    for node in list(graph.nodes):
        if node.op == "output":
            graph.erase_node(node)

    with graph.inserting_after():
        graph.output(node_to_output)


def set_dict_outputs_by_name(graph: Graph, outputs: Iterable[str]) -> None:
    """Modify the given Graph by adding a new node which collects multiple outputs in a dict. This
    new node will then become the output of the graph.
    """
    # Find the named nodes to be the args to a new output node
    nodes_to_output = get_nodes_by_name(graph, outputs)

    # Remove all preexisting outputs
    for node in list(graph.nodes):
        if node.op == "output":
            graph.erase_node(node)

    with graph.inserting_after():
        # Create a new node which collects the outputs into a dict
        collector_node = graph.call_function(
            the_function=dict, kwargs={name: node for name, node in zip(outputs, nodes_to_output)}
        )
        # Set the new 'collector' node as the output of the graph
        graph.output(collector_node)

    graph.eliminate_dead_code()


def set_inputs_and_output_by_name(graph: Graph, inputs: Iterable[str], output: str) -> None:
    """Set the inputs and output of a graph to the nodes of the given name(s)."""
    _assert_no_common_names(inputs, [output])
    # It's important that we do the following operations in the correct order. Setting the output
    # defines what code is 'alive' or 'dead', and removing dead code is necessary before calling
    # _set_inputs_by_name, otherwise we will get an error trying to remove the existing inputs.
    _set_output_by_name(graph, output)
    graph.eliminate_dead_code()
    _set_inputs_by_name(graph, inputs)


def get_subgraph(graph_module: GraphModule, inputs: Iterable[str], output: str) -> GraphModule:
    """Extract a subgraph from a GraphModule by specifying the input and output nodes by name. The
    returned GraphModule shares attributes/submodules/parameters with the original, but the graph
    is a new object. This allows for things like freeze(get_subgraph(module)) to freeze some subset
    of the model.
    """
    new_module = _copy_module_new_graph(graph_module, name="Sub" + graph_module.__class__.__name__)
    set_inputs_and_output_by_name(new_module.graph, inputs, output)
    new_module.recompile()
    return new_module


def _assert_no_common_names(names1: Iterable[str], names2: Iterable[str]) -> None:
    common_names = set(names1) & set(names2)
    if common_names:
        raise ValueError(f"Redundant: {common_names}")


def prefix_all_nodes(graph: Graph, prefix: str) -> Graph:
    # For details on opcodes see https://pytorch.org/docs/stable/fx.html#Node
    for node in graph.nodes:
        # All nodes get renamed
        node.name = f"{prefix}_{node.name}"

        # Other node attributes are handled on a per-opcode basis
        match node.op:
            case "placeholder":
                node.target = node.name
            case "get_attr" | "call_module" | "call_method":
                # target is the name of a module attribute, which were all renamed above. Note that
                # the convention for modules and submodules is a "." join, while the convention
                # for nodes is a "_" join.
                node.target = f"{prefix}.{node.target}"
            case "call_function" | "output":
                # If any node.args or node.kwargs are themselves references to other nodes, they
                # will have been prefixed already, so we don't need to do anything here.
                pass
            case _:
                assert_never(node.op)
    return graph


def stitch_graphs(
    named_modules: dict[str, nn.Module],
    rewire_layers_from_to: dict[str, str],
    input_names: Iterable[str],
    output_name: str,
) -> GraphModule:
    # Get a GraphModule version of each module
    named_graph_modules = {
        k: symbolic_trace(v) if not isinstance(v, GraphModule) else v
        for k, v in named_modules.items()
    }

    # Rename all nodes in the modules' respective graphs and copy them into a big new graph
    new_graph = Graph()
    for name, module in named_graph_modules.items():
        new_graph.graph_copy(prefix_all_nodes(deepcopy(module.graph), name), {})

    # Rewire the specified nodes
    from_nodes = get_nodes_by_name(new_graph, rewire_layers_from_to.keys())
    to_nodes = get_nodes_by_name(new_graph, rewire_layers_from_to.values())
    for from_node, to_node in zip(from_nodes, to_nodes):
        to_node.replace_all_uses_with(from_node)

    # Create the new GraphModule with attributes/submodules from the original modules. Wrapping
    # the named_modules in a ModuleDict means that the names of the named_modules will act as dot
    # prefixes. For instance, if named_modules is {"a": moduleA, "b": moduleB}, then the 'root'
    # will have attributes "a" and "b" which are the respective modules. Attributes in the graph
    # were prefixed in the call to prefix_all_nodes() above to reflect this.
    new_module = GraphModule(
        root=nn.ModuleDict(named_modules),
        graph=new_graph,
        class_name="_".join(named_modules.keys()),
    )

    # Set inputs and outputs. Note that this must happen after creating the new GraphModule because
    # creating the new module has a necessary side, effect of populating new_graph.owning_module,
    # which is needed by set_inputs_and_output_by_name.
    set_inputs_and_output_by_name(new_graph, input_names, output_name)

    # Recompile the module because outputs/inputs changed
    new_module.recompile()

    return new_module


def squash_all_conv_batchnorm_pairs(graph_module: GraphModule) -> GraphModule:
    """Squash all conv-batchnorm pairs in a model. Returns a new model with parameters/attributes
    shared with the original model *except* for the new conv layers.

    Args:
        graph_module: The model to squash.

    Returns:
        The modified model.
    """
    new_module = _copy_module_new_graph(
        graph_module, name="Squashed" + graph_module.__class__.__name__
    )

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
        conv_module, bn_module = new_module.get_submodule(conv.target), new_module.get_submodule(bn.target)
        squashed_conv = squash_conv_batchnorm(conv_module, bn_module)
        new_module.add_submodule(squashed_name, squashed_conv)
        with new_module.graph.inserting_before(conv):
            new_node = new_module.graph.call_module(
                squashed_name, args=conv.args, kwargs=conv.kwargs
            )
            bn.replace_all_uses_with(new_node)

    # Post-surgery, clean up the graph. This will remove all unused nodes, so any conv/bn nodes
    # that we squashed end up removed but only if they are no longer used in any other part of the
    # graph.
    new_module.graph.eliminate_dead_code()
    new_module.recompile()

    return new_module


def to_dot(graph: Graph) -> pydot.Dot:
    dot = pydot.Dot()
    for node in graph.nodes:
        dot.add_node(pydot.Node(node.name))
        for user in node.users:
            dot.add_edge(pydot.Edge(node.name, user.name))
    return dot


def step_through_call(graph_module: GraphModule, context={}) -> Any:
    """Step through a call to a GraphModule, printing the name of each node and the shape of each
    tensor as it passes through the node."""

    def _get_arg(arg: Any):
        if isinstance(arg, Node):
            return context[arg.name]
        return arg

    for node in graph_module.graph.nodes:
        match node.op:
            case "placeholder":
                assert node.name in context, f"Missing input {node.name}"
            case "get_attr":
                context[node.name] = getattr(graph_module, node.target)
            case "call_module":
                module = graph_module.get_submodule(node.target)
                args = [_get_arg(arg) for arg in node.args]
                kwargs = {k: _get_arg(v) for k, v in node.kwargs.items()}
                context[node.name] = module(*args, **kwargs)
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
