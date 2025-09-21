import argparse
import os

import torch

from src.mlp.models import VariationalPredictor
from src.mlp.datasets import get_mnist_dataloaders
from src.mlp.trainers.methods.vib import train_vib
from src.mlp.datasets.mnist.utils import MNIST_MERGE_GROUP


TSNE_CONFIG = {
    "merge_group": MNIST_MERGE_GROUP,
    "coarse_mapping": {i: f"group {i}" for i in range(len(MNIST_MERGE_GROUP))},
    "fine_mapping": None
}


def main(args):
    dataloaders = get_mnist_dataloaders(root=args.data_dir, batch_size=args.batch_size)

    print(f"train: {len(dataloaders[0].dataset)} samples")
    print(f"valid: {len(dataloaders[1].dataset)} samples")
    print(f"test: {len(dataloaders[2].dataset)} samples")

    ckpt_dir = os.path.join(args.ckpt_dir, f"mnist{'_' + args.bound_z_by if args.bound_z_by is not None else ''}")

    vib_model = VariationalPredictor(num_classes=len(MNIST_MERGE_GROUP), input_dim=28 * 28, is_posthoc=False, bound_z_by=args.bound_z_by)
    vib_model = vib_model.to(args.device)

    train_vib(vib_model, train_loader=dataloaders[0], valid_loader=dataloaders[1],
              ckpt_dir=ckpt_dir, x_key=None, y_key=None,
              dataset_name="mnist",
              beta=args.beta, device=args.device,
              bound_z_by=args.bound_z_by, is_posthoc=False,
              tsne_config=TSNE_CONFIG, epochs=args.epochs,
              lr=args.lr, if_existing_ckpt=args.if_existing_ckpt, y_grain="coarse")
    vib_model_best_path = os.path.join(args.ckpt_dir,
                                       f"mnist{'_' + args.bound_z_by if args.bound_z_by is not None else ''}",
                                       "vib",
                                       f"beta_{args.beta:.1E}",
                                       "vib_coarse_best.pth")
    vib_model_best_ckpt = torch.load(vib_model_best_path, weights_only=False)
    vib_model.load_state_dict(vib_model_best_ckpt["model_state_dict"])
    for param in vib_model.parameters():
        param.requires_grad = False
    vib_model = vib_model.to("cpu")
    torch.cuda.empty_cache()

    posthoc_model = VariationalPredictor(num_classes=10, input_dim=28 * 28, is_posthoc=True, encoder=vib_model.encoder, bound_z_by=args.bound_z_by)
    posthoc_model = posthoc_model.to(args.device)
    train_vib(posthoc_model, train_loader=dataloaders[0], valid_loader=dataloaders[1],
              ckpt_dir=ckpt_dir, x_key=None, y_key=None,
              dataset_name="mnist",
              beta=args.beta, device=args.device,
              bound_z_by=args.bound_z_by, is_posthoc=True,
              tsne_config=TSNE_CONFIG, epochs=args.epochs,
              lr=args.lr, if_existing_ckpt=args.if_existing_ckpt, y_grain="fine")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VIB for MNIST")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/data/MNIST/",
                        help="Directory for MNIST data")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/checkpoints/mnist",
                        help="Directory for saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training")
    parser.add_argument("--batch_per_epoch", type=int, default=100,
                        help="Number of batches per epoch (for epoch definition)")
    parser.add_argument("--bound_z_by", type=str, default=None, choices=[None, "tanh", "sigmoid"],
                        help="Bound latent space z by a function")
    parser.add_argument("--beta", type=float, default=1e-3, help="Beta value for VIB loss")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for optimizer")
    parser.add_argument("--if_existing_ckpt", type=str, default="resume", choices=["resume", "reset", "pass"],
                        help="Action if existing checkpoint is found: resume, reset, or pass")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()
    print(f"Using device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    main(args)
