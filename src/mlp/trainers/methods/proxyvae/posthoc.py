import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import wandb

from src.mlp.models import VariationalPredictor
# from src.mlp.trainers.utils import get_confusion_matrix_heatmap_as_nparray


WANDB_PROJECT = "ProxyVAE"
WANDB_ENTITY = "garyeechung-vanderbilt-university"


def train_model(model: VariationalPredictor, train_loader,
                x_key, y_key, optimizer, loss_fn, device, y_grain="coarse"):
    model.train()

    total_losses = 0.0

    y_true_all = []
    y_pred_all = []
    for batch in train_loader:
        x = batch[0].float().to(device)
        if y_grain == "coarse":
            y = batch[1].argmax(dim=-1).long().to(device)
        elif y_grain == "fine":
            y = batch[2].argmax(dim=-1).long().to(device)
        else:
            raise ValueError(f"y_grain {y_grain} not recognized.")
        y_true_all.append(y.cpu().numpy())

        optimizer.zero_grad()
        y_pred, _, _ = model(x)
        y_pred_all.append(y_pred.detach().cpu().numpy().argmax(axis=-1))

        # suppose loss_fn is CrossEntropyLoss, y and y_pred are both float
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_losses += loss.item()
    avg_loss = total_losses / len(train_loader)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    accuracy = np.mean(y_true_all == y_pred_all)
    return avg_loss, accuracy


def evaluate_model(model: VariationalPredictor, valid_loader,
                   x_key, y_key, loss_fn, device, y_grain="coarse"):
    model.eval()

    total_losses = 0.0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in valid_loader:
            x = batch[0].float().to(device)
            if y_grain == "coarse":
                y = batch[1].argmax(dim=-1).long().to(device)
            elif y_grain == "fine":
                y = batch[2].argmax(dim=-1).long().to(device)
            else:
                raise ValueError(f"y_grain {y_grain} not recognized.")
            y_true_all.append(y.cpu().numpy())

            y_pred, _, _ = model(x)
            y_pred_all.append(y_pred.detach().cpu().numpy().argmax(axis=-1))

            # suppose loss_fn is CrossEntropyLoss, y and y_pred are both float
            loss = loss_fn(y_pred, y)
            total_losses += loss.item()
    avg_loss = total_losses / len(valid_loader)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    accuracy = np.mean(y_true_all == y_pred_all)
    return avg_loss, accuracy


def train_posthoc_predictor(model: VariationalPredictor,
                            train_loader, valid_loader,
                            ckpt_dir: str,
                            x_key: str, y_key: str,
                            beta1: float, beta2: float, device: str,
                            dataset_name: str, bound_z_by: str,
                            epochs: int = 500, lr: float = 5e-4,
                            if_existing_ckpt: str = "resume", y_grain="coarse"):
    batch_size = next(iter(train_loader))[0].shape[0]
    batch_per_epoch = len(train_loader)
    config = {
        "model": f"posthoc_{y_key}",
        "target": y_key,
        "beta1": beta1,
        "beta2": beta2,
        "lr": lr,
        "batch_size": batch_size,
        "batch_per_epoch": batch_per_epoch,
        "bound_z_by": bound_z_by,
    }
    ckpt_dir = os.path.join(ckpt_dir, "proxyvae", f"beta1_{beta1:.1E}", f"beta2_{beta2:.1E}")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    ckpt_path = os.path.join(ckpt_dir, f"posthoc_{y_key}.pth")
    ckpt_path_best = os.path.join(ckpt_dir, f"posthoc_{y_key}_best.pth")

    if os.path.exists(ckpt_path) and if_existing_ckpt == "pass":
        checkpoint = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = checkpoint["epoch"]
        return model
    elif os.path.exists(ckpt_path) and if_existing_ckpt == "resume":
        checkpoint = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_valid_loss = checkpoint["best_valid_loss"]

        ckpt_epoch = checkpoint["epoch"]
        if ckpt_epoch >= epochs:
            return model
        else:
            epochs -= ckpt_epoch
    elif os.path.exists(ckpt_path) and if_existing_ckpt == "replace":
        os.remove(ckpt_path)
        best_valid_loss = float("inf")
        ckpt_epoch = 0
    else:
        best_valid_loss = float("inf")
        ckpt_epoch = 0

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               group=f"{dataset_name}",
               name=f"posthoc_{y_key}_beta1_{beta1:.1E}_beta2_{beta2:.1E}",
               config=config)

    model = model.to(device)

    for epoch in tqdm(range(ckpt_epoch + 1, ckpt_epoch + epochs + 1)):
        train_loss, train_acc = train_model(
            model=model, train_loader=train_loader,
            x_key=x_key, y_key=y_key, y_grain=y_grain,
            optimizer=optimizer, loss_fn=loss_fn, device=device)
        valid_loss, valid_acc = evaluate_model(
            model=model, valid_loader=valid_loader,
            x_key=x_key, y_key=y_key, y_grain=y_grain,
            loss_fn=loss_fn, device=device)

        wandb.log({
            "train/ce_loss": train_loss,
            "train/accuracy": train_acc,
            "valid/ce_loss": valid_loss,
            "valid/accuracy": valid_acc,
        })

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
            }, ckpt_path_best)

    latest_ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_loss": best_valid_loss,
    }
    torch.save(latest_ckpt, ckpt_path)
    wandb.finish()
    return model
