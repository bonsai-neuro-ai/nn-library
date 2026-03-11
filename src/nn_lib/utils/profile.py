import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn import min


def profile_training_throughput_torch_dataloader(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: str = "cuda",
    pin_memory: bool = True,
    non_blocking: bool = True,
    num_batches: int = 100,
) -> float:
    """
    Profiles the training throughput of a given model on a specified dataset.

    Args:
        model (nn.Module): The neural network model to be profiled.
        dataset (Dataset): The dataset to use for profiling.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of worker threads for data loading.
        device (str): The device to run the model on ('cuda' or 'cpu').
        pin_memory (bool): Whether to pin memory in DataLoader.
        non_blocking (bool): Whether to use non-blocking transfers to device.
        num_batches (int): Number of batches to process for profiling.

    Returns:
        float: Time taken per item in seconds.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    model = model.to(device).train()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()

        if i >= num_batches - 1:
            break

    end_time.record()
    torch.cuda.synchronize()

    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert ms to s
    time_per_item = elapsed_time / (num_batches * batch_size)

    return time_per_item


def optimize_for_throughput_torch_dataloader(
    model: nn.Module,
    dataset: Dataset,
    batch_sizes: list,
    num_workers_list: list,
    device: str = "cuda",
    pin_memory: bool = True,
    non_blocking: bool = True,
    num_batches: int = 100,
) -> dict:
    """
    Finds the optimal batch size and number of workers for maximum throughput.

    Args:
        model (nn.Module): The neural network model to be profiled.
        dataset (Dataset): The dataset to use for profiling.
        batch_sizes (list): List of batch sizes to test.
        num_workers_list (list): List of number of workers to test.
        device (str): The device to run the model on ('cuda' or 'cpu').
        pin_memory (bool): Whether to pin memory in DataLoader.
        non_blocking (bool): Whether to use non-blocking transfers to device.
        num_batches (int): Number of batches to process for profiling.

    Returns:
        dict: Dictionary containing the best configuration and its throughput.
    """
    best_time_per_item = float("inf")
    best_config = {}

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            time_per_item = profile_training_throughput_torch_dataloader(
                model,
                dataset,
                batch_size,
                num_workers,
                device,
                pin_memory,
                non_blocking,
                num_batches,
            )

            if time_per_item < best_time_per_item:
                best_time_per_item = time_per_item
                best_config = {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "time_per_item": time_per_item,
                }

    return best_config


__all__ = [
    "profile_training_throughput_torch_dataloader",
    "optimize_for_throughput_torch_dataloader",
]
