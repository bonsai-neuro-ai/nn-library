import jsonargparse
import mlflow
import pandas as pd
import numpy as np
from nn_lib.env import add_parser as add_env_parser
from nn_lib.models import get_model_graph
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from typing import Optional
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.testing as npt


def load_data(expt_name, tracking_uri, similarity_method="LinearCKA"):
    """Return DF where each row is some model1.layer1 compared to some model2.layer2."""
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
    plt.hlines([target_stress], 1, len(pair_dist), colors="k", linestyles="--")
    plt.vlines([low], 0, target_stress, colors="k", linestyles="--")
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


def plot_model_paths(xyz: np.ndarray, columns, dims_xy, ax=None):
    ax = ax or plt.gca()
    points_by_model = defaultdict(list)
    layer_names_by_model = defaultdict(list)
    for i, (model, layer) in enumerate(columns):
        points_by_model[model].append(xyz[i, dims_xy])
        layer_names_by_model[model].append(layer)

    for model in points_by_model:
        points = np.array(points_by_model[model])
        resort_idx = argsort_layers(model, layer_names_by_model[model])
        ax.plot(*points[resort_idx, :].T, label=model, marker=".")


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

    # Embed the distances into a lower-dimensional space
    dim = find_embedding_dim(dist, threshold=0.99)
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
                plot_model_paths(xyz, tbl.columns, dims_xy=(j, i), ax=ax[i, j])
                ax[i, j].set_xlabel(f"dim {j+1}")
                ax[i, j].set_ylabel(f"dim {i+1}")
                ax[i, j].axis("equal")
                ax[i, j].grid()
    ax[0,0].legend()
    fig.tight_layout()
    plt.show()

    # 3D plot of the first 3 PCs
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    plot_model_paths(xyz, tbl.columns, dims_xy=(0, 1, 2), ax=ax)
    ax.legend()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    plt.show()
