import argparse
import os

import torch

from src.cnn.models import VariationalPredictor
from src.cnn.datasets import get_cifar100_dataloaders
from src.cnn.trainers.infobottleneck import train_infobottleneck


def main(args):
    dataloaders = get_cifar100_dataloaders(args.data_dir, batch_size=args.batch_size, val_ratio=0.1)
    print(f"train: {len(dataloaders[0].dataset)} samples")
    print(f"valid: {len(dataloaders[1].dataset)} samples")
    print(f"test: {len(dataloaders[2].dataset)} samples")

    ckpt_dir = os.path.join(args.ckpt_dir, f"{args.backbone}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}")
    group_model = VariationalPredictor(num_classes=20, backbone=args.backbone,
                                       image_channels=3, image_size=[32, 32],
                                       latent_dim=256, base_channels=4,
                                       is_posthoc=False, bound_z_by=args.bound_z_by)
    group_model = group_model.to(args.device)
    group_model = train_infobottleneck(model=group_model,
                                       train_loader=dataloaders[0],
                                       valid_loader=dataloaders[1],
                                       ckpt_dir=ckpt_dir,
                                       x_key="image",
                                       y_key="coarse_label",
                                       dataset_name="cifar100",
                                       backbone=args.backbone,
                                       beta=args.beta,
                                       bootstrap=False,
                                       bound_z_by=args.bound_z_by,
                                       device=args.device,
                                       epochs=args.epochs,
                                       lr=args.lr,
                                       if_existing_ckpt="resume")
    group_model = group_model.to("cpu")
    torch.cuda.empty_cache()

    group_model_best_path = os.path.join(args.ckpt_dir,
                                         f"{args.backbone}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}",
                                         "infobottleneck",
                                         f"beta_{args.beta:.1E}",
                                         "infobottleneck_coarse_label_best.pth")
    group_model_best_ckpt = torch.load(group_model_best_path, weights_only=False)
    group_model.load_state_dict(group_model_best_ckpt["model_state_dict"])
    for param in group_model.parameters():
        param.requires_grad = False

    class_model = VariationalPredictor(num_classes=100, encoder=group_model.encoder,
                                       image_channels=3, image_size=[32, 32],
                                       latent_dim=256, base_channels=4,
                                       is_posthoc=True, bound_z_by=args.bound_z_by)
    class_model = class_model.to(args.device)
    class_model = train_infobottleneck(model=class_model,
                                       train_loader=dataloaders[0],
                                       valid_loader=dataloaders[1],
                                       ckpt_dir=ckpt_dir,
                                       x_key="image",
                                       y_key="fine_label",
                                       dataset_name="cifar100",
                                       backbone=args.backbone,
                                       beta=args.beta,
                                       bootstrap=False,
                                       bound_z_by=args.bound_z_by,
                                       device=args.device,
                                       epochs=args.epochs,
                                       lr=args.lr,
                                       if_existing_ckpt="resume")
    class_model = class_model.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train InfoBottleneck for CIFAR-100")
    parser.add_argument("--data_dir", type=str, default="/home/chungk1/Repositories/InvaRep/data/CIFAR",
                        help="Directory for CIFAR-100 data")
    parser.add_argument("--ckpt_dir", type=str, default="/home/chungk1/Repositories/InvaRep/checkpoints/cifar100",
                        help="Directory to save checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone architecture")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for InfoBottleneck loss")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--if_existing_ckpt", type=str, default="resume",
                        choices=["resume", "replace", "pass"],
                        help="What to do if an existing checkpoint is found")
    parser.add_argument("--bound_z_by", type=str, default=None,
                        choices=[None, "tanh", "standardization", "normalization"],
                        help="How to bound the latent space z")
    args = parser.parse_args()
    main(args)
    print("Training completed.")
