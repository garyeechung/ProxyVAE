import argparse
import os

import torch

from src.mlp.models import ConditionalVAE
from src.mlp.datasets import get_mnist_dataloaders
from src.mlp.datasets.mnist.utils import MNIST_MERGE_GROUP
from src.mlp.trainers.methods.proxyvae import train_cvae


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

    cvae = ConditionalVAE(num_classes=len(MNIST_MERGE_GROUP), input_dim=28 * 28, bound_z_by=args.bound_z_by)
    cvae = cvae.to(args.device)
    ckpt_dir = os.path.join(args.ckpt_dir, f"mnist{'_' + args.bound_z_by if args.bound_z_by is not None else ''}")

    train_cvae(cvae, train_loader=dataloaders[0], valid_loader=dataloaders[1],
               ckpt_dir=ckpt_dir,
               x_key=None,
               y_key=None,
               dataset_name="mnist",
               beta1=args.beta1,
               bound_z_by=args.bound_z_by,
               device=args.device,
               epochs=args.epochs,
               lr=args.lr,
               if_existing_ckpt=args.if_existing_ckpt,
               tsne_config=TSNE_CONFIG,
               comparison_fn=None,
               y_grain="coarse")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE for MNIST")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/data/MNIST/",
                        help="Directory for MNIST data")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/checkpoints/mnist",
                        help="Directory to save checkpoints")
    parser.add_argument("--beta1", type=float, default=1.0, help="Beta1 parameter for CVAE loss")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--batch_per_epoch", type=int, default=100,
                        help="Number of batches per epoch")
    parser.add_argument("--if_existing_ckpt", type=str, default="resume",
                        choices=["resume", "replace", "pass"],
                        help="What to do if an existing checkpoint is found")
    parser.add_argument("--bound_z_by", type=str, default=None,
                        choices=[None, "tanh", "standardization", "normalization"],
                        help="How to bound the latent space z")
    args = parser.parse_args()

    main(args)
