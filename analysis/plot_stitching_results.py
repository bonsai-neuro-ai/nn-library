import jsonargparse
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from nn_lib.analysis.stitching import StitchingStage
import numpy.testing as npt


def load_data(expt_name, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    df = mlflow.search_runs(experiment_names=[expt_name], filter_string="status='FINISHED'")
    print(*df.columns, sep="\n\t")
    print(f"Loaded {len(df)} runs")
    print("Columns:")
    return df


def sanity_check_results(df):
    by_stage = {
        stage: df[df["params.stage"] == str(stage)].set_index(
            ["params.model/init_args/layer1", "params.model/init_args/layer2"]
        )
        for stage in StitchingStage
    }
    npt.assert_allclose(
        by_stage[StitchingStage.RANDOM_INIT]["metrics.model1-test_loss"],
        by_stage[StitchingStage.REGRESSION_INIT]["metrics.model1-test_loss"],
    )
    npt.assert_allclose(
        by_stage[StitchingStage.RANDOM_INIT]["metrics.model1-test_loss"],
        by_stage[StitchingStage.TRAIN_STITCHING_LAYER]["metrics.model1-test_loss"],
    )
    npt.assert_allclose(
        by_stage[StitchingStage.RANDOM_INIT]["metrics.model1-test_loss"],
        by_stage[StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM]["metrics.model1-test_loss"],
    )
    npt.assert_allclose(
        by_stage[StitchingStage.RANDOM_INIT]["metrics.model2-test_loss"],
        by_stage[StitchingStage.REGRESSION_INIT]["metrics.model2-test_loss"],
    )
    npt.assert_allclose(
        by_stage[StitchingStage.RANDOM_INIT]["metrics.model2-test_loss"],
        by_stage[StitchingStage.TRAIN_STITCHING_LAYER]["metrics.model2-test_loss"],
    )
    # Include a small buffer in this loss comparison because the process is a little noisy
    npt.assert_array_less(
        by_stage[StitchingStage.RANDOM_INIT]["metrics.model2-test_loss"],
        by_stage[StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM]["metrics.model2-test_loss"] + .01,
        err_msg="Expected model2 loss to get *worse* after fine-tuning downstream model2 layers for stitching",
    )


def plot_loss_per_stage(df):
    fig = plt.figure(figsize=(10, 8))
    g = sns.boxplot(data=df, x="params.stage", y="metrics.stitched-test_loss", order=StitchingStage)
    g.set_yscale("log")
    g.set_xticks(
        range(4),
        labels=[str(stage).lower() for stage in StitchingStage],
        rotation=45,
        horizontalalignment="right",
    )
    fig.tight_layout()
    plt.show()


def visualize_test_loss(df):
    for stage in df["params.stage"].unique():
        matrix = df[df["params.stage"] == stage].pivot_table(
            index="params.model/init_args/layer1",
            columns="params.model/init_args/layer2",
            values="metrics.stitched-test_loss",
        )
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=False, vmin=0)
        # sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=6)
        plt.title(f"Test loss per model 1 layer vs model 2 layer in stage {stage.lower()}")
        fig.tight_layout()
        plt.show()


def visualize_between_stage_change_in_loss(df):
    matrix_regression = df[df["params.stage"] == str(StitchingStage.REGRESSION_INIT)].pivot_table(
        index="params.model/init_args/layer1",
        columns="params.model/init_args/layer2",
        values="metrics.stitched-test_loss",
    )
    matrix_stitching = df[
        df["params.stage"] == str(StitchingStage.TRAIN_STITCHING_LAYER)
    ].pivot_table(
        index="params.model/init_args/layer1",
        columns="params.model/init_args/layer2",
        values="metrics.stitched-test_loss",
    )
    matrix_all = df[
        df["params.stage"] == str(StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM)
    ].pivot_table(
        index="params.model/init_args/layer1",
        columns="params.model/init_args/layer2",
        values="metrics.stitched-test_loss",
    )

    loss_diff = matrix_stitching - matrix_regression
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(loss_diff, center=0, cmap="coolwarm")
    # sns.heatmap(loss_diff, annot=True, fmt=".2f", center=0, cmap="coolwarm")
    plt.title("Improvement in loss from training the stitching layer vs regression init")
    fig.tight_layout()
    plt.show()

    loss_diff = matrix_all - matrix_stitching
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(loss_diff, center=0, cmap="coolwarm")
#     sns.heatmap(loss_diff, annot=True, fmt=".2f", center=0, cmap="coolwarm")
    plt.title("Improvement in loss from fine-tuning all layers vs just the stitching layer")
    fig.tight_layout()
    plt.show()


def visualize_training_time(df):
    df_train_stitching = df[df["params.stage"] == str(StitchingStage.TRAIN_STITCHING_LAYER)]
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_train_stitching.pivot_table(
            index="params.model/init_args/layer1",
            columns="params.model/init_args/layer2",
            values="metrics.stitched-epoch",
        ).astype(int),
        annot=True,
        fmt="d",
        vmin=0,
    )
    plt.title("Epochs to train stitching layer")
    fig.tight_layout()
    plt.show()

    df_train_all = df[
        df["params.stage"] == str(StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM)
    ]
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_train_all.pivot_table(
            index="params.model/init_args/layer1",
            columns="params.model/init_args/layer2",
            values="metrics.stitched-epoch",
        ).astype(int),
        annot=True,
        fmt="d",
        vmin=0,
    )
    plt.title("Epochs to train all layers")
    fig.tight_layout()
    plt.show()


def main(args):
    df = load_data(args.expt_name, args.tracking_uri)

    # Architectures are specified by architecture (depth and width)
    # TODO - and how trained
    arch_key_1 = [
        "params.model/init_args/model1/init_args/depth",
        "params.model/init_args/model1/init_args/width",
    ]
    arch_key_2 = [
        "params.model/init_args/model2/init_args/depth",
        "params.model/init_args/model2/init_args/width",
    ]

    for key1, group1 in df.groupby(arch_key_1):
        for key2, group2 in group1.groupby(arch_key_2):
            print("KEY1", key1, "KEY2", key2)
            sanity_check_results(group2)
            plot_loss_per_stage(group2)
            visualize_test_loss(group2)
            visualize_between_stage_change_in_loss(group2)
            visualize_training_time(group2)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--expt_name", type=str, required=True, help="Name of the MLflow experiment"
    )
    parser.add_argument(
        "--tracking_uri", type=str, required=True, help="URI of the MLflow tracking server"
    )
    args = parser.parse_args()

    main(args)
