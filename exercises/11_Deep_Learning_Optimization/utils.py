import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch


def set_seeds():
    seed = 0
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_default_device():
    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def eval_model(model, epoch, results, device, loss_fn, eval_fn, dataloaders, verbose=False):
    """Evaluate the model using a loss_fn and eval_fn on all dataloaders."""

    if verbose:
        print(f" * Eval model at the end of Epoch {epoch}")

    model.eval()

    for split, dataloader in dataloaders.items():
        loss, acc = eval_dataset(model, device, loss_fn, eval_fn, dataloader)
        results[split]["loss"].append(loss)
        results[split]["accuracy"].append(acc)

    return results


def eval_dataset(model, device, loss_fn, eval_fn, dataloader):
    """Evaluate the model on a single dataloader."""

    with torch.no_grad():
        agg_loss, agg_eval_metric = 0.0, 0.0
        num_elements = 0
        for data in dataloader:
            # Get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # Forward pass
            outputs = model(inputs)
            _loss = loss_fn(outputs, labels).item()

            # compute eval metric
            _eval_metric = eval_fn(outputs, labels)

            # Normalize
            num_elements_in_batch = len(labels)
            num_elements += num_elements_in_batch
            agg_loss += _loss * num_elements_in_batch
            agg_eval_metric += _eval_metric * num_elements_in_batch

    loss = agg_loss / num_elements
    acc = agg_eval_metric / num_elements

    return loss, acc


def accuracy_fn(logits, labels):
    """Compute the accuracy for given logits and labels."""

    predicted = torch.argmax(logits, 1)
    correct_predictions = (predicted == labels).sum().item()

    return correct_predictions / len(labels)

def visualize_results(results, verbose=False):
    """Visualize the results by showing the loss and accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 4])

    for split in ["train", "validation", "test"]:
        ax1.plot(results[split]["loss"], label=split.capitalize())

    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    for split in ["train", "validation", "test"]:
        ax2.plot(results[split]["accuracy"], label=split.capitalize())

    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    
    if verbose:
        print(
            "The best achieved train/valid/tests accuracies are: ",
            max(results["train"]["accuracy"]), "/",
            max(results["validation"]["accuracy"]), "/",
            max(results["test"]["accuracy"]),
        )
        
def get_logpath(suffix=""):
    """Create a logpath and return it.

    Args:
        suffix (str, optional): suffix to add to the output. Defaults to "".

    Returns:
        str: Path to the logfile (output of Cockpit).
    """
    save_dir = "cockpit_logfiles"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"cockpit_logfile")
    return log_path