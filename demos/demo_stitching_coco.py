import io
import os
from pathlib import Path

import dotenv
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.transforms import v2

dotenv.load_dotenv()
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from tqdm.auto import tqdm

from nn_lib.models import (
    GraphModulePlus,
    Interpolate2d,
    RegressableConv2d,
    get_pretrained_model,
)
from nn_lib.optim import LRFinder
from nn_lib.optim.lr_finder import UnstableLREstimate
from nn_lib.utils import conv2d_shape_inverse, frozen

# Same category list as torchvision/references/segmentation/train.py
VOC_CATS = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]


class CocoAsSemanticSeg(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.lut = torch.zeros(91, dtype=torch.long)  # non-VOC categories -> background
        self.lut[torch.tensor(VOC_CATS)] = torch.arange(len(VOC_CATS))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, tgt = self.ds[i]
        sem = torch.zeros(img.shape[-2:], dtype=torch.long)
        if "masks" in tgt and len(tgt["labels"]) > 0:  # some images have no annotations
            cls = self.lut[tgt["labels"]]
            m = tgt["masks"].bool() & (cls > 0)[:, None, None]
            sem = (m.long() * cls[:, None, None]).amax(dim=0)
            sem[m.sum(0) > 1] = 255  # overlapping instances -> ignore
        return img, sem


def vis_dataset_sample(dataset: CocoDetection):
    example = dataset[101]
    plt.imshow(example[0].permute(1, 2, 0) / 4 + 0.5)  # un-normalize the image for visualization
    plt.axis("off")
    plt.show()


# We can also visualize the model as a graph
def display_model_graph(mdl: GraphModulePlus, dpi: int = 200):
    image = mdl.to_dot().create_png(prog="dot")
    with io.BytesIO(image) as f:
        image = mpimg.imread(f)
    plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("TempGraph.png")
    plt.show()


def is_rank_zero() -> bool:
    """True on the main process, or when not running distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # Before init_process_group (or under torchrun/mp.spawn setup), fall back to env
    return int(os.environ.get("RANK", 0)) == 0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CoCo dataset and use a simple default transform.
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(520),
            v2.CenterCrop(448),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    coco_images = Path(os.getenv("DATA_ROOT")) / "coco" / "images" / "val2017"
    coco_annots = Path(os.getenv("DATA_ROOT")) / "coco" / "annotations" / "instances_val2017.json"
    dataset = CocoAsSemanticSeg(
        wrap_dataset_for_transforms_v2(
            CocoDetection(coco_images, coco_annots, transforms=transforms),
            target_keys=("boxes", "labels", "masks"),
        )
    )

    # Model names must be recognized by torchvision.models.get_model
    modelA = get_pretrained_model("resnet18")
    modelB = get_pretrained_model("fcn_resnet50")

    # To do stitching or other 'model surgery', we need to convert from standard nn.Module objects
    # to torch's GraphModule objects. This is done by 'tracing' the input->output flow of the model
    modelA = (
        GraphModulePlus.new_from_trace(modelA).squash_all_conv_batchnorm_pairs().to(device).eval()
    )
    modelB = (
        GraphModulePlus.new_from_trace(modelB)
        .squash_all_conv_batchnorm_pairs()
        .set_output("interpolate")  # this is out; or interpolate_1 for aux
        .to(device)
        .eval()
    )

    # Print out the layer names of the model; these are the names of the nodes in the computation
    # graph
    if is_rank_zero():
        modelA.graph.print_tabular()
        display_model_graph(modelA)
        display_model_graph(modelB)

    # Create a hybrid stitched model.
    layerA = "add_3"
    layerB = "add_5"
    layerB_noop = modelB.insert_noop(layerB)

    # Step (1): Get 'subgraph' models which go input -> desired layer. Then, look at the output
    # shape of these sub-models to determine the shape of the tensors at the desired layers.
    # Note: if the two models were trained on different datasets or expect different input sizes,
    # this needs to be modified.
    embedding_getterA = GraphModulePlus.new_from_copy(modelA).extract_subgraph(output=layerA)
    embedding_getterB = GraphModulePlus.new_from_copy(modelB).extract_subgraph(output=layerB)

    with torch.no_grad():
        dummy_input = dataset[0][0].to(device).unsqueeze(0)
        dummy_A = embedding_getterA(dummy_input)
        dummy_B = embedding_getterB(dummy_input)

    # Step (2): Create a Conv1x1StitchingLayer from the shapes of the tensors at the desired layers
    conv_params = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
    }
    stitching_layer = nn.Sequential(
        Interpolate2d(
            size=conv2d_shape_inverse(dummy_B.shape[-2:], **conv_params),
            mode="bilinear",
        ),
        RegressableConv2d(
            in_channels=dummy_A.shape[1], out_channels=dummy_B.shape[1], **conv_params
        ),
    )

    # Step (3): model merging surgery. Note the types: modelA and modelB are already GraphModules, but
    # stitching_layer is a regular nn.Module. Setting auto_trace=False keeps it as a regular nn.Module,
    # which is necessary for us to call init_by_regression() on it later.
    modelAB = (
        GraphModulePlus.new_from_merge(
            modules={
                "modelA": modelA,
                "stitching_layer": stitching_layer,
                "modelB": modelB,
            },
            rewire_inputs={
                "stitching_layer": "modelA_" + layerA,
                f"modelB_{layerB_noop}": "stitching_layer",
            },
            auto_trace=False,
        )
        .to(device)
        .eval()
    )

    # Let's also visualize the stitched model as an image
    display_model_graph(modelAB)

    # Sanity-check that we can run modelB and modelAB in the same way
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    images, targets = next(iter(data_loader))
    print(f"Sanity-checking on a single test batch containing {len(images)} images")
    images = images.to(device)
    targets = targets.to(device)
    modelA = modelA.to(device)
    modelB = modelB.to(device)
    modelAB = modelAB.to(device)

    def quick_run_and_check(model, im, ta, name):
        with torch.no_grad():
            output = model(im)

        # See https://github.com/pytorch/vision/blob/main/references/segmentation/train.py
        loss = nn.functional.cross_entropy(output, ta, ignore_index=255)
        print(name, "loss:", loss.item())

    quick_run_and_check(modelB, images, targets, "ModelB")
    # we expect this one to be bad because the stitching layer was only randomly initialized
    quick_run_and_check(modelAB, images, targets, "ModelAB")

    # Record parameters before any training so that we can sanity-check that only the correct
    # things are changing. The .clone() is necessary so we get a snapshot of the parameters at
    # this point in time, rather than a reference to the parameters which will be updated later.
    paramsA = {k: v.clone() for k, v in modelA.named_parameters()}
    paramsB = {k: v.clone() for k, v in modelB.named_parameters()}
    paramsAB = {k: v.clone() for k, v in modelAB.named_parameters()}

    # Now that we have a functioning stitched model, let's update the stitching layer. First way
    # to do this is with the regression-based method. Note that this will in general be better if
    # we use a bigger batch.
    print("=== DOING REGRESSION INIT ===")
    num_regression_batches = 10
    for b, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device), targets.to(device)
        regression_from = modelAB.stitching_layer[0](embedding_getterA(images))
        regression_to = embedding_getterB(images)
        modelAB.stitching_layer[1].init_by_regression(
            regression_from,
            regression_to,
            batched=True,
            final_batch=b == num_regression_batches - 1,
        )
        if b == num_regression_batches - 1:
            break

    # Assert that no parameters changed *except* for stitched_model.stitching_layer
    for k, v in modelA.named_parameters():
        assert torch.allclose(v, paramsA[k])
    for k, v in modelB.named_parameters():
        assert torch.allclose(v, paramsB[k])
    for k, v in modelAB.named_parameters():
        if k.startswith("stitching_layer"):
            assert not torch.allclose(v, paramsAB[k])
        else:
            assert torch.allclose(v, paramsAB[k])

    # Take a new parameter shapshot since some changed
    paramsAB = {k: v.clone() for k, v in modelAB.named_parameters()}

    # Re-run and see if it's improved
    quick_run_and_check(modelB, images, targets, "ModelB")
    quick_run_and_check(modelAB, images, targets, "ModelAB")

    # For fine-tuning the stitching layer while freezing the rest of the model, I've created a
    # model freezing context manager. This also illustrates an important design choice: the
    # parameters of stitched_model are *shared* with the original modelA and modelB models. Two
    # important consequences of this: (1) we can freeze modelA while training stitched_model,
    # and the parameters of modelA will not be updated; (2) if we update the parameters of
    # stitched_model, the parameters of modelA and modelB will also be updated. We need to always
    # be careful to make copies of models if we want to avoid this.

    # Before training, try to discover the optimal learning rate
    modelAB.stitching_layer.train()
    tmp_optimizer = torch.optim.Adam(modelAB.stitching_layer.parameters(), lr=1e-8)
    lr_finder = LRFinder(
        modelAB,
        tmp_optimizer,
        criterion=lambda inputs, outputs, targets: nn.functional.cross_entropy(
            outputs, targets, ignore_index=255
        ),
        device=device,
    )
    lr_finder.range_test(data_loader, 1e-8, 0.1, num_iter=100, step_mode="exp")
    try:
        lr = lr_finder.suggestion()
    except UnstableLREstimate:
        lr = 1e-4
    del tmp_optimizer
    optimizer = torch.optim.Adam(modelAB.stitching_layer.parameters(), lr=lr)

    # Assert that no parameters changed during LR optimization
    for k, v in modelA.named_parameters():
        assert torch.allclose(v, paramsA[k])
    for k, v in modelB.named_parameters():
        assert torch.allclose(v, paramsB[k])
    for k, v in modelAB.named_parameters():
        assert torch.allclose(v, paramsAB[k])

    history = []
    # To train stitching layer AND downstream model, just remove 'modelB' from the list of frozen
    # models
    with frozen(modelA, modelB):
        # Train for 100 steps or 1 epoch, whichever comes first
        for step, (images, targets) in tqdm(
            enumerate(data_loader), total=100, desc="Train Stitching Layer"
        ):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            output = modelAB(images)
            # This is task loss, but could be updated to be soft-labels to optimize match to model B
            loss = nn.functional.cross_entropy(output, targets, ignore_index=255)
            history.append(loss.item())
            loss.backward()
            optimizer.step()

            step += 1
            if step == 100:
                break

    if is_rank_zero():
        plt.plot(history)
        plt.xlabel("Training Step")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Training the Stitching Layer by itself")
        plt.show()

    # Assert that no parameters changed *except* for stitched_model.stitching_layer
    for k, v in modelA.named_parameters():
        assert torch.allclose(v, paramsA[k])
    for k, v in modelB.named_parameters():
        assert torch.allclose(v, paramsB[k])
    for k, v in modelAB.named_parameters():
        if k.startswith("stitching_layer"):
            assert not torch.allclose(v, paramsAB[k])
        else:
            assert torch.allclose(v, paramsAB[k])

    # Re-run and see if it's improved
    quick_run_and_check(modelB, images, targets, "ModelB")
    quick_run_and_check(modelAB, images, targets, "ModelAB")
