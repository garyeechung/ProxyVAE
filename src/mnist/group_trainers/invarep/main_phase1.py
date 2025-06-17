import argparse
import torch

from src.mnist.models import CVAE
from .cvae import train_cvae, MERGE_GROUP


def main(args):
    model = CVAE(num_classes=len(MERGE_GROUP))
    device = args.device
    train_cvae(model=model, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
               beta1=args.beta1, device=device, merge_group=MERGE_GROUP,
               batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
               return_each_batch=True, replace_existing_ckpt=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE for MNIST Group Similarity")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/mnist/group_similar", help="Directory to save checkpoints")
    parser.add_argument("--beta1", type=float, default=1.0, help="Beta parameter for CVAE loss")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    args = parser.parse_args()

    main(args)
