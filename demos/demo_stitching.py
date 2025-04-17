import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm

import nn_lib.models.utils
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import (
    get_pretrained_model,
    GraphModulePlus,
    RegressableConv2d,
    Interpolate2d,
    conv2d_shape_inverse,
)
from nn_lib.optim import LRFinder
from nn_lib.optim.lr_finder import UnstableLREstimate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset, pointing root_dir to the location on the server where we keep shared datasets
data_module = ImageNetDataModule(root_dir="/data/datasets/")
# DataModule is a concept from pytorch lightning. For performance reasons, data modules are lazy
# and wait to load the data until we actually ask for it. We need to tell the data module to
# actually run its setup routines. There's room for API improvement here, especially the unexpected
# behavior where prepare_data() has a side-effect of setting up the default transforms, which are
# then required by setup().
data_module.prepare_data()
data_module.setup("test")

# Model names must be recognized by torchvision.models.get_model
modelA = get_pretrained_model("resnet18")
modelB = get_pretrained_model("resnet34")

# To do stitching or other 'model surgery', we need to convert from standard nn.Module objects
# to torch's GraphModule objects. This is done by 'tracing' the input->output flow of the model
modelA = GraphModulePlus.new_from_trace(modelA).squash_all_conv_batchnorm_pairs().to(device).eval()
modelB = GraphModulePlus.new_from_trace(modelB).squash_all_conv_batchnorm_pairs().to(device).eval()

# Print out the layer names of the model; these are the names of the nodes in the computation graph
modelA.graph.print_tabular()


# We can also visualize the model as a graph
def display_model_graph(mdl, dpi=200):
    image = mdl.to_dot().create_png(prog="dot")
    with io.BytesIO(image) as f:
        image = mpimg.imread(f)
    plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("TempGrap.png")
    plt.show()


display_model_graph(modelA)
display_model_graph(modelB)

# Create a hybrid stitched model.
layerA = "add_3"
layerB = "add_5"
layerB_noop = modelB.insert_noop(layerB)

# Step (1): Get 'subgraph' models which go input -> desired layer. Then, look at the output shape
# of these sub-models to determine the shape of the tensors at the desired layers. Note: if the two
# models were trained on different datasets or expect different input sizes, this needs to be
# modified.
dummy_input = torch.zeros(data_module.shape, device=device).unsqueeze(0)
embedding_getterA = GraphModulePlus.new_from_copy(modelA).extract_subgraph(output=layerA)
embedding_getterB = GraphModulePlus.new_from_copy(modelB).extract_subgraph(output=layerB)

with torch.no_grad():
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
    Interpolate2d(size=conv2d_shape_inverse(dummy_B.shape[-2:], **conv_params), mode="bilinear"),
    RegressableConv2d(in_channels=dummy_A.shape[1], out_channels=dummy_B.shape[1], **conv_params),
)

# Step (3): model merging surgery. Note the types: modelA and modelB are already GraphModules, but
# stitching_layer is a regular nn.Module. Setting auto_trace=False keeps it as a regular nn.Module,
# which is necessary for us to call init_by_regression() on it later.
modelAB = (
    GraphModulePlus.new_from_merge(
        modules={"modelA": modelA, "stitching_layer": stitching_layer, "modelB": modelB},
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

# Sanity-check that we can run all 3 models
data_loader = data_module.test_dataloader(batch_size=100, num_workers=4)
images, labels = next(iter(data_loader))
images, labels = images.to(device), labels.to(device)
print(f"Sanity-checking on a single test batch containing {len(images)} images")
modelA = modelA.to(device)
modelB = modelB.to(device)
modelAB = modelAB.to(device)


def quick_run_and_check(model, images, labels, name):
    with torch.no_grad():
        output = model(images)
    print(f"{name}: {torch.sum(torch.argmax(output, dim=1) == labels).item()} / {len(labels)}")


quick_run_and_check(modelA, images, labels, "ModelA")
quick_run_and_check(modelB, images, labels, "ModelB")
# we expect this one to be bad because the stitching layer was only randomly initialized
quick_run_and_check(modelAB, images, labels, "ModelAB")

# Record parameters before any training so that we can sanity-check that only the correct things
# are changing. The .clone() is necessary so we get a snapshot of the parameters at this point in
# time, rather than a reference to the parameters which will be updated later.
paramsA = {k: v.clone() for k, v in modelA.named_parameters()}
paramsB = {k: v.clone() for k, v in modelB.named_parameters()}
paramsAB = {k: v.clone() for k, v in modelAB.named_parameters()}

data_module.setup("fit")
train_dataloader = data_module.train_dataloader()

# Now that we have a functioning stitched model, let's update the stitching layer. First way to do
# this is with the regression-based method. Note that this will in general be better if we use a
# bigger batch.
print("=== DOING REGRESSION INIT ===")
num_regression_batches = 10
for b, (images, labels) in enumerate(data_loader):
    images, labels = images.to(device), labels.to(device)
    regression_from = modelAB.stitching_layer[0](embedding_getterA(images))
    regression_to = embedding_getterB(images)
    modelAB.stitching_layer[1].init_by_regression(
        regression_from, regression_to, batched=True, final_batch=b == num_regression_batches - 1
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
quick_run_and_check(modelA, images, labels, "ModelA")
quick_run_and_check(modelB, images, labels, "ModelB")
quick_run_and_check(modelAB, images, labels, "ModelAB")

# For fine-tuning the stitching layer while freezing the rest of the model, I've created a model
# freezing context manager. This also illustrates an important design choice: the parameters of
# stitched_model are *shared* with the original modelA and modelB models. Two important consequences of
# this: (1) we can freeze modelA while training stitched_model, and the parameters of modelA will not be
# updated; (2) if we update the parameters of stitched_model, the parameters of modelA and modelB will
# also be updated. We need to always be careful to make copies of models if we want to avoid this.

# Before training, try to discover the optimal learning rate
modelAB.stitching_layer.train()
tmp_optimizer = torch.optim.Adam(modelAB.stitching_layer.parameters(), lr=1e-8)
lr_finder = LRFinder(modelAB, tmp_optimizer, criterion=nn.CrossEntropyLoss(), device=device)
lr_finder.range_test(train_dataloader, 1e-8, 0.1, num_iter=100, step_mode="exp")
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
# To train stitching layer AND downstream model, just remove 'modelB' from the list of frozen models
with nn_lib.models.utils.frozen(modelA, modelB):
    # Train for 100 steps or 1 epoch, whichever comes first
    for step, (im, la) in tqdm(
        enumerate(train_dataloader), total=100, desc="Train Stitching Layer"
    ):
        optimizer.zero_grad()
        output = modelAB(images)
        # This is task loss, but could be updated to be soft-labels to optimize match to model B
        loss = torch.nn.functional.cross_entropy(output, labels)
        history.append(loss.item())
        loss.backward()
        optimizer.step()

        step += 1
        if step == 100:
            break

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
quick_run_and_check(modelA, images, labels, "ModelA")
quick_run_and_check(modelB, images, labels, "ModelB")
quick_run_and_check(modelAB, images, labels, "ModelAB")
