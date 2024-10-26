import torch
import pydot
import warnings
from torch.fx import GraphModule, Graph, Node
from typing import Iterable, Generator, assert_never
from nn_lib.models.utils import squash_conv_batchnorm


__all__ = [
    "GraphModule",
    "Graph",
    "Node",
    "get_nodes_by_name",
    "get_subgraph",
    "set_inputs_and_output_by_name",
    "stitch_graphs",
    "squash_all_conv_batchnorm_pairs",
    "to_dot",
]


def get_nodes_by_name(
    graph: Graph,
    names: str | Iterable[str],
) -> Generator[Node, None, None]:
    """Get nodes from a graph by name. The name argument may be a string or an iterable of strings.

    Raises a ValueError after iteration is complete if not all requested names were present in
    the graph.
    """
    if isinstance(names, str):
        names = [names]
    names = set(names)
    for node in graph.nodes:
        if node.name in names:
            yield node
            names.remove(node.name)
    if names:
        raise ValueError("Not all nodes are present in the graph:", names)


def _copy_module_new_graph(graph_module: GraphModule) -> GraphModule:
    new_graph = Graph()
    output_node = new_graph.graph_copy(graph_module.graph, {})
    if output_node is not None:
        new_graph.output(output_node)
    return GraphModule(root=graph_module, graph=new_graph)


def _set_inputs_by_name(graph: Graph, inputs: Iterable[str]) -> None:
    """Set the inputs of a graph by finding nodes of the given name(s) and replacing them with
    placeholders."""
    # For each named input, erase any existing nodes of the same name and replace them with a
    # new placeholder node.
    for node in get_nodes_by_name(graph, inputs):
        with graph.inserting_before(node):
            new_placeholder = graph.placeholder("tmp_new_input")
            node.replace_all_uses_with(new_placeholder)
            graph.erase_node(node)
            new_placeholder.name = node.name

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
    node_to_output = next(get_nodes_by_name(graph, output))

    # Remove all preexisting outputs
    for node in list(graph.nodes):
        if node.op == "output":
            graph.erase_node(node)

    with graph.inserting_after():
        graph.output(node_to_output)


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
    new_module = _copy_module_new_graph(graph_module)
    set_inputs_and_output_by_name(new_module.graph, inputs, output)
    new_module.delete_all_unused_submodules()
    new_module.recompile()
    return new_module


def _assert_no_common_names(names1: Iterable[str], names2: Iterable[str]) -> None:
    common_names = set(names1) & set(names2)
    if common_names:
        raise ValueError(f"Redundant: {common_names}")


def _prefix_all_attributes(graph_module: GraphModule, prefix: str) -> GraphModule:
    # Submodules are handled recursively. By packaging all submodules into a ModuleDict and then
    # storing them in an attribute with the name "{prefix}", an attribute like "model.fc" will
    # become "model.{prefix}.fc".
    submodules = torch.nn.ModuleDict(dict(graph_module.named_children()))
    for name, submodule in submodules.items():
        graph_module.delete_submodule(name)
    graph_module.add_module(prefix, submodules)
    return graph_module


def _prefix_all_nodes(graph: Graph, prefix: str) -> Graph:
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
    named_modules: dict[str, GraphModule],
    rewire_layers_from_to: dict[str, str],
    input_names: Iterable[str],
    output_name: str,
) -> GraphModule:
    # Rename all nodes in the modules' respective graphs
    new_graph = Graph()
    for name, module in named_modules.items():
        new_graph.graph_copy(_prefix_all_nodes(module.graph, name), {})

    # Rewire the specified nodes
    from_nodes = get_nodes_by_name(new_graph, rewire_layers_from_to.keys())
    to_nodes = get_nodes_by_name(new_graph, rewire_layers_from_to.values())
    for from_node, to_node in zip(from_nodes, to_nodes):
        to_node.replace_all_uses_with(from_node)

    # Create the new GraphModule
    new_module = GraphModule(root=torch.nn.ModuleDict(named_modules), graph=new_graph)

    # Set inputs and outputs. Note that this must happen after creating the new GraphModule because
    # creating the new module has a necessary side, effect of populating new_graph.owning_module,
    # which is needed by set_inputs_and_output_by_name.
    set_inputs_and_output_by_name(new_graph, input_names, output_name)

    # Strip unused attributes so that the new module only has parameters/submodules corresponding
    # to the parts of the graph that are actually used.
    new_module.delete_all_unused_submodules()

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
    new_module = _copy_module_new_graph(graph_module)

    # Find all conv-batchnorm pairs
    # TODO: handle functional calls like F.conv2d; currently we assume all convs and batchnorms
    #  are called as modules.
    conv_bn_pairs = []
    for node in new_module.graph.nodes:
        if node.op == "call_module" and isinstance(
            new_module.get_submodule(node.target), torch.nn.Conv2d
        ):
            for user in node.users:
                if user.op == "call_module" and isinstance(
                    new_module.get_submodule(user.target), torch.nn.BatchNorm2d
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
        squashed_conv = squash_conv_batchnorm(
            new_module.get_submodule(conv.target), new_module.get_submodule(bn.target)
        )
        new_module.add_submodule(squashed_name, squashed_conv)
        with new_module.graph.inserting_before(conv):
            new_node = new_module.graph.call_module(squashed_name, args=conv.args, kwargs=conv.kwargs)
            bn.replace_all_uses_with(new_node)

    # Post-surgery, clean up the graph. This will remove all unused nodes, so any conv/bn nodes
    # that we squashed end up removed but only if they are no longer used in any other part of the
    # graph.
    new_module.graph.eliminate_dead_code()
    new_module.delete_all_unused_submodules()
    new_module.recompile()

    return new_module


def to_dot(graph: Graph) -> pydot.Dot:
    dot = pydot.Dot()
    for node in graph.nodes:
        dot.add_node(pydot.Node(node.name))
        for user in node.users:
            dot.add_edge(pydot.Edge(node.name, user.name))
    return dot
