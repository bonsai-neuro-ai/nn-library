import jsonargparse
import mlflow
import pandas as pd
import numpy as np
from nn_lib.env import add_parser as add_env_parser
from nn_lib.models import get_model_graph
from nn_lib.models.graph_utils import get_topology_for_subset_of_layers
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from typing import Optional
from collections import defaultdict
import matplotlib.pyplot as plt


def load_data(expt_name, tracking_uri, similarity_method="LinearCKA"):
    """Return DF where each row is some model1.layer1 compared to some model2.layer2."""
    # TODO - allow restriction to certain models/layers here
    mlflow.set_tracking_uri(tracking_uri)
    # By searching for runs where metrics."{method}" > 0, we filter out parent runs (where metric
    # will be nan)
    return mlflow.search_runs(
        experiment_names=[expt_name],
        filter_string=f"status='FINISHED' AND metrics.\"{similarity_method}\" > 0",
    )


def pairwise_similarity_pivot_table(
    df: pd.DataFrame, metric: str, symmetric: bool = False
) -> pd.DataFrame:
    # Create a pivot table where each row is a model1.layer1 and each column is a model2.layer2
    # The value is the similarity between the two layers
    pair_sim = df.pivot_table(
        index=("params.model1", "params.layer1"),
        columns=("params.model2", "params.layer2"),
        values=f"metrics.{metric}",
    )

    # In the symmetric case, [i, j] being NaN can pull values from [j, i]
    if symmetric:
        pair_sim = pair_sim.combine_first(pair_sim.T)

    return pair_sim


def find_embedding_dim(pair_dist: np.ndarray, threshold: float = 0.95):
    """Find the minimum embedding dimension such that the explained embedding stress is 'threshold'
    percent of the way from 1D to ND.
    """
    stress_1 = MDS(n_components=1, dissimilarity="precomputed").fit(pair_dist).stress_
    stress_n = MDS(n_components=len(pair_dist), dissimilarity="precomputed").fit(pair_dist).stress_
    record = [(1, stress_1), (len(pair_dist), stress_n)]
    target_stress = (1 - threshold) * (stress_1 - stress_n) + stress_n
    # Use binary search to narrow down the embedding dimension
    low, high = 1, pair_dist.shape[0]
    while low < high:
        mid = (low + high) // 2
        this_stress = MDS(n_components=mid, dissimilarity="precomputed").fit(pair_dist).stress_
        record.append((mid, this_stress))
        if this_stress < target_stress:
            high = mid
        else:
            low = mid + 1

    record = sorted(record)
    plt.hlines([target_stress], 1, len(pair_dist), colors="k", linestyles="dashed")
    plt.vlines([low], stress_n, stress_1, colors="k", linestyles="dashed")
    plt.plot(*zip(*record), marker=".")
    plt.xlabel("dim")
    plt.ylabel("stress")
    plt.show()

    return low


def embed(pair_dist: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    """Embed pair_dist into a lower-dimensional space using MDS."""
    if dim is None:
        dim = find_embedding_dim(pair_dist)
    mds = MDS(n_components=dim, dissimilarity="precomputed")
    return mds.fit_transform(pair_dist)


def argsort_layers(model_name, layers):
    # TODO - compute topology including branches. For now, just using topological sort 'as if'
    #  idx[i] is connected to idx[i+1]
    indices = []
    for node in get_model_graph(model_name).nodes:
        if node.name in layers:
            indices.append(layers.index(node.name))
    return indices


def get_model_topologies(columns):
    topologies = {}
    for model in set(model for model, _ in columns):
        layers = [layer for m, layer in columns if m == model]
        # TODO - reduce dependency on instantiating a full model here just to inspect its topology
        # TODO - aux layers not showing up?
        topologies[model] = get_topology_for_subset_of_layers(
            get_model_graph(model, squash=True), layers
        )
    return topologies


def plot_model_paths(
    xyz: np.ndarray, columns, dims_xy=(0, 1), topologies=None, ax=None, cmap="tab10"
):
    ax = ax or plt.gca()
    points_by_layer_by_model = defaultdict(dict)
    for i, (model, layer) in enumerate(columns):
        points_by_layer_by_model[model][layer] = xyz[i, dims_xy]

    if topologies is None:
        topologies = get_model_topologies(columns)

    cm = plt.get_cmap(cmap)
    for i, model in enumerate(points_by_layer_by_model):
        layers = list(points_by_layer_by_model[model].keys())
        points = np.array(list(points_by_layer_by_model[model].values()))
        ax.plot(*points[0], marker=".", color=cm(i), label=model)
        for j in range(1, len(points)):
            ax.plot(*points[j], marker=".", color=cm(i))
            # ax.text(*points[j], layers[j], fontsize=8, color=cm(i))

        for layer_name, connections in topologies[model].items():
            for connected_layer_name in connections:
                xy0 = points[layers.index(layer_name)]
                xy1 = points[layers.index(connected_layer_name)]
                ax.plot(*zip(xy0, xy1), color=cm(i))


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local/env.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    parser.add_argument("--metric", type=str, default="LinearCKA")
    add_env_parser(parser)
    args = parser.instantiate_classes(parser.parse_args())

    df = load_data(args.expt_name, args.env.mlflow_tracking_uri, similarity_method=args.metric)
    tbl = pairwise_similarity_pivot_table(df, metric=args.metric, symmetric=True)

    dist = np.arccos(tbl.to_numpy())
    assert np.allclose(dist, dist.T, atol=1e-3)
    dist = (dist + dist.T) / 2

    # Precompute layer:layer topology for each model for the purposes of drawing.
    topologies = get_model_topologies(tbl.columns)

    # Embed the distances into a lower-dimensional space
    # dim = find_embedding_dim(dist, threshold=0.99)
    dim = 11
    print("Using embedding dimension:", dim)
    mds_xyz = embed(dist, dim)
    xyz = PCA(n_components=dim).fit_transform(mds_xyz)

    # Plot grid of the first 5 PCs
    fig, ax = plt.subplots(5, 5, figsize=(15, 15))
    for i in range(5):
        for j in range(5):
            if j >= i:
                ax[i, j].remove()
            else:
                plot_model_paths(
                    xyz, tbl.columns, dims_xy=(j, i), ax=ax[i, j], topologies=topologies
                )
                ax[i, j].set_xlabel(f"dim {j+1}")
                ax[i, j].set_ylabel(f"dim {i+1}")
                ax[i, j].axis("equal")
                ax[i, j].grid()
    ax[0, 0].legend()
    fig.tight_layout()
    plt.show()

    # 3D plot of the first 3 PCs
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    plot_model_paths(xyz, tbl.columns, dims_xy=(0, 1, 2), ax=ax, topologies=topologies)
    ax.legend()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    plt.show()
