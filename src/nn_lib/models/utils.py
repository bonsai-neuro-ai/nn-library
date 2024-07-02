import pydot
from nn_lib.models.graph import GraphType


class Add:
    def __call__(self, x, y):
        return x + y


class Identity:
    def __call__(self, x):
        return x


def graph2dot(network_graph: GraphType) -> pydot.Graph:
    edges = []
    for layer, (_, parents) in network_graph.items():
        for pa in parents:
            edges.append((pa, layer))
    return pydot.graph_from_edges(edges, directed=True)
