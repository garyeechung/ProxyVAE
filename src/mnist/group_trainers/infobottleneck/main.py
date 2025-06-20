import argparse
import torch

from src.mnist.models import InfoBottleneckClassifier, ProxyRep2Label
from .model import train_infobottleneck_groupifier
from .posthoc import train_infobottleneck_posthoc
from src.mnist.datasets import MERGE_GROUP


def main(args):
    model = InfoBottleneckClassifier(nb_labels=len(MERGE_GROUP))
    device = args.device
    train_infobottleneck_groupifier(model=model, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                                    beta=args.beta, device=device, batch_size=args.batch_size,
                                    lr=args.lr, epochs=args.epochs, merge_group=MERGE_GROUP,
                                    replace_existing_ckpt=False)

    for param in model.parameters():
        param.requires_grad = False

    posthoc_model = ProxyRep2Label(autoencoder=model, reparameterize=True, nb_labels=10)
    train_infobottleneck_posthoc(posthoc_model=posthoc_model, data_dir=args.data_dir,
                                 ckpt_dir=args.ckpt_dir, beta=args.beta, device=device,
                                 batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                                 replace_existing_ckpt=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train InfoBottleneck Classifier")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/mnist/group_similar", help="Directory to save checkpoints")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for InfoBottleneck loss")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    args = parser.parse_args()

    main(args)
