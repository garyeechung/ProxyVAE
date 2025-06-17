import argparse
import torch

from src.mnist.models import CVAE, InvariantVariationalAutoEncoder, ProxyRep2Label, ProxyRep2InvarRep
from .cvae import MERGE_GROUP
from .proxyvae import train_proxyvae
from .posthoc_group import train_posthoc_group
from .posthoc_class import train_posthoc_class
from .proxy2invarep import train_proxy2invarep


def main(args):

    # Load the CVAE model
    cvae = CVAE(num_classes=len(MERGE_GROUP))
    device = args.device
    cvae_ckpt = f"{args.ckpt_dir}/invarep/beta1_{args.beta1}/cvae.pth"
    cvae.load_state_dict(torch.load(cvae_ckpt))
    for param in cvae.parameters():
        param.requires_grad = False

    # Second phase: Train the Invariant Variational Autoencoder (ProxyVAE)
    proxyvae = InvariantVariationalAutoEncoder(cvae=cvae)
    proxyvae.to(device)
    train_proxyvae(model=proxyvae, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                   beta1=args.beta1, beta2=args.beta2, device=device,
                   batch_size=args.batch_size, epochs=args.epochs,
                   lr=args.lr, return_each_batch=True,
                   replace_existing_ckpt=False)
    for param in proxyvae.parameters():
        param.requires_grad = False

    # Post-hoc proxy to invariant representation
    proxy2invarep = ProxyRep2InvarRep(autoencoder=proxyvae, reparameterize=True)
    proxy2invarep = proxy2invarep.to(device)
    train_proxy2invarep(model=proxy2invarep, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                        beta1=args.beta1, beta2=args.beta2, device=device,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        return_each_batch=True, replace_existing_ckpt=False)

    # Post-hoc groupifier
    posthoc_groupifier = ProxyRep2Label(autoencoder=proxyvae,
                                        reparameterize=True,
                                        nb_labels=len(MERGE_GROUP))
    posthoc_groupifier = posthoc_groupifier.to(device)
    train_posthoc_group(model=posthoc_groupifier, data_dir=args.data_dir,
                        ckpt_dir=args.ckpt_dir, beta1=args.beta1, beta2=args.beta2,
                        device=device, batch_size=args.batch_size, epochs=int(args.epochs * 0.6),
                        lr=args.lr, return_each_batch=True,
                        replace_existing_ckpt=False)

    # Post-hoc classifier
    posthoc_classifier = ProxyRep2Label(autoencoder=proxyvae,
                                        reparameterize=True,
                                        nb_labels=10)
    posthoc_classifier = posthoc_classifier.to(device)
    train_posthoc_class(model=posthoc_classifier, data_dir=args.data_dir,
                        ckpt_dir=args.ckpt_dir, beta1=args.beta1, beta2=args.beta2,
                        device=device, batch_size=args.batch_size, epochs=int(args.epochs * 0.6),
                        lr=args.lr, return_each_batch=True,
                        replace_existing_ckpt=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProxyVAE for MNIST Group Similarity")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/mnist/group_similar", help="Directory to save checkpoints")
    parser.add_argument("--beta1", type=float, default=1.0, help="Beta1 parameter for CVAE loss")
    parser.add_argument("--beta2", type=float, default=1.0, help="Beta2 parameter for ProxyVAE loss")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    args = parser.parse_args()

    main(args)
