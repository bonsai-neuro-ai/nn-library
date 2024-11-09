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
from pydot import graph_from_edges
from io import BytesIO
from PIL import Image
import warnings
import matplotlib.pyplot as plt


def hotfix_model2color(model_name):
    cmap = plt.get_cmap("tab20")
    match model_name:
        case "fcn_resnet50":
            return cmap(0)
        case "deeplabv3_resnet50":
            return cmap(1)
        case "resnet18":
            return cmap(2)
        case "resnet34":
            return tuple((np.array(cmap(2)) + np.array(cmap(3))) / 2)
        case "resnet50":
            return cmap(3)
        case "vit_b_16":
            return cmap(4)
        case "vit_b_32":
            return cmap(5)
        case _:
            warnings.warn("No hand-coded color for model, using default.")
            return cmap(6)


def load_data(expt_name, tracking_uri, similarity_method="LinearCKA") -> pd.DataFrame:
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
    fig, ax = plt.subplots()
    ax.plot([1, len(pair_dist)], [target_stress] * 2, "--k")
    ax.plot([low] * 2, [stress_1, stress_n], "--k")
    ax.plot(*zip(*record), marker=".")
    ax.set_xlabel("dim")
    ax.set_ylabel("stress")
    ax.set_yscale("log")
    fig.tight_layout()
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
    for node in get_model_graph(model_name, squash=True).nodes:
        if node.name in layers:
            indices.append(layers.index(node.name))
    # Sanity check that all layers were found
    missed = set(layers) - set(layers[i] for i in indices)
    if missed:
        warnings.warn(f"Failed to find layers {missed} in model {model_name}")
        # Just append the missed layers to the end
        indices.extend(layers.index(m) for m in missed)
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
    xyz: np.ndarray,
    columns,
    dims_xy=(0, 1),
    topologies=None,
    ax=None,
    label: bool = True,
):
    ax = ax or plt.gca()
    points_by_layer_by_model = defaultdict(dict)
    for i, (model, layer) in enumerate(columns):
        points_by_layer_by_model[model][layer] = xyz[i, dims_xy]

    if topologies is None:
        topologies = get_model_topologies(columns)

    for model in points_by_layer_by_model:
        layers = list(points_by_layer_by_model[model].keys())
        points = np.array(list(points_by_layer_by_model[model].values()))
        ax.plot(
            *points[0], marker=".", color=hotfix_model2color(model), label=model if label else None
        )
        for j in range(1, len(points)):
            ax.plot(*points[j], marker=".", color=hotfix_model2color(model))
            # ax.text(*points[j], layers[j], fontsize=8, color=hotfix_model2color(model))

        for layer_name, connections in topologies[model].items():
            for connected_layer_name in connections:
                xy0 = points[layers.index(layer_name)]
                xy1 = points[layers.index(connected_layer_name)]
                ax.plot(*zip(xy0, xy1), color=hotfix_model2color(model))


def make_topology_image(topology, savename):
    edges = []
    for layer, connections in topology.items():
        edges.extend((layer, connected_layer) for connected_layer in connections)
    graph = graph_from_edges(edges)
    with open(savename, "wb") as f:
        f.write(graph.create_png(prog="dot"))


def plot_path_pcs(pca_xyz, columns, n=5, topologies=None, fig=None):
    fig = fig or plt.figure()
    ax = fig.subplots(n - 1, n - 1, sharex=True, sharey=True)
    for i in range(n - 1):
        for j in range(n - 1):
            if j > i:
                ax[i, j].remove()
            else:
                plot_model_paths(
                    pca_xyz,
                    columns,
                    dims_xy=(j, i + 1),
                    ax=ax[i, j],
                    topologies=topologies,
                    label=i == 0 and j == 0,
                )
                ax[i, j].set_xlabel(f"dim {j+1}")
                ax[i, j].set_ylabel(f"dim {i+2}")
                ax[i, j].grid()
    plt.figlegend(
        *ax[0, 0].get_legend_handles_labels(),
        loc="upper right",
        fancybox=True,
        shadow=True,
    )
    fig.tight_layout()
    plt.show()


def plot_layer_layer_distances(pair_distances, columns):
    # Columns are (model, layer) tuples. Keep models in the same order, but sort layers within each
    # model using argsort_layers():
    indices_for_each_model = defaultdict(list)
    for i, (model, _) in enumerate(columns):
        indices_for_each_model[model].append(i)

    # Assert that no layers are 'interleaved' between models; i.e. if model1 has layers 0, 1, 2,
    # then model2 should start at 3, and we should find no further model1 indices after that.
    last_range_end = -1
    boundaries = []
    for model, indices in indices_for_each_model.items():
        assert (
            min(indices) > last_range_end
        ), "Sanity-check failed: layers from different models are interleaved?"
        last_range_end = max(indices)
        boundaries.append(last_range_end)

    # Find global sort order for layers
    model_start, sort_indices = 0, []
    for model, indices in indices_for_each_model.items():
        layers_this_model = [columns[i][1] for i in indices]
        sort_indices.extend(model_start + i for i in argsort_layers(model, layers_this_model))
        model_start += len(indices)

    pair_distances = pair_distances[sort_indices, :][:, sort_indices]

    plt.figure(figsize=(10, 10))
    plt.imshow(pair_distances, cmap="viridis", vmin=0, vmax=np.pi / 2, interpolation="nearest")
    for b in boundaries[:-1]:
        plt.plot([b + 0.5] * 2, [-0.5, len(pair_distances) - 0.5], color="k")
        plt.plot([-0.5, len(pair_distances) - 0.5], [b + 0.5] * 2, color="k")
    # Indicate model names in between boundaries
    boundaries = [-0.5] + [b + 0.5 for b in boundaries]
    tick_locations = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
    plt.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labeltop=True,
        labelbottom=False,
        labelleft=True,
        labelright=False,
        labelrotation=45,
    )
    plt.xticks(tick_locations, labels=list(indices_for_each_model.keys()))
    plt.yticks(tick_locations, labels=list(indices_for_each_model.keys()))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local/env.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    parser.add_argument("--metric", type=str, default="LinearCKA")
    add_env_parser(parser)
    args = parser.instantiate_classes(parser.parse_args())

    print("Loading data...")
    df = load_data(args.expt_name, args.env.mlflow_tracking_uri, similarity_method=args.metric)
    tbl = pairwise_similarity_pivot_table(df, metric=args.metric, symmetric=True)

    dist = np.arccos(tbl.to_numpy())
    assert np.allclose(dist, dist.T, atol=1e-3)
    dist = (dist + dist.T) / 2

    plot_layer_layer_distances(dist, tbl.columns)

    subset_of_interest = [
        ("fcn_resnet50", "add_15"),
        ("deeplabv3_resnet50", "add_15"),
        ("resnet18", "add_7"),
        ("resnet34", "add_15"),
        ("resnet50", "add_15"),
        ("vit_b_16", "add_24"),
        ("vit_b_32", "add_24"),
    ]
    indices = [i for i, col in enumerate(tbl.columns) if col in subset_of_interest]
    assert len(indices) == len(subset_of_interest)
    tbl2 = tbl.iloc[indices].iloc[:, indices]
    plot_layer_layer_distances(np.arccos((tbl2.to_numpy() + tbl2.to_numpy().T) / 2), tbl2.columns)

    exit(0)

    # Precompute layer:layer topology for each model for the purposes of drawing.
    print("Getting model topologies...")
    topologies = get_model_topologies(tbl.columns)
    for mdl, topo in topologies.items():
        make_topology_image(topo, f"topology_{mdl}.png")

    # Embed the distances into a lower-dimensional space
    print("Finding best dimensionality and doing MDS...")
    dim = find_embedding_dim(dist, threshold=0.99)
    print("Using embedding dimension:", dim)
    mds_xyz = embed(dist, dim)
    pca = PCA(n_components=dim)
    pca.fit(mds_xyz)
    xyz = pca.transform(mds_xyz)

    print("Plotting...")

    plt.figure()
    plt.plot(pca.explained_variance_ratio_, marker=".")
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.show()

    # Plot grid of the first 5 PCs
    plot_path_pcs(xyz, tbl.columns, n=3, topologies=topologies, fig=plt.figure(figsize=(6, 6)))

    # 3D plot of the first 3 PCs
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    plot_model_paths(xyz, tbl.columns, dims_xy=(0, 1, 2), ax=ax, topologies=topologies)
    ax.legend()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    plt.show()
