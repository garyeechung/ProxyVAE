import argparse
import os

import pandas as pd
import torch

from src.adni.models import ConditionalVAE
from src.adni.datasets import get_adni_dataloaders
from .cvae import train_cvae


def main(args):
    df = pd.read_csv(os.path.join(args.data_dir, "adni_data.csv"))
    dataloaders = get_adni_dataloaders(
        df, data_dir=os.path.join(args.data_dir, "FA_rigid_MNI_1mm"),
        targets=["manufacturer_id", "model_type_id"],
        batch_size=args.batch_size, include_mappable_site_empty=False,
        cache_type="persistent", bootstrap=args.bootstrap,
        num_workers=0, batch_per_epoch=args.batch_per_epoch,
        spatial_size=args.spatial_size,
        slice_range_from_center=args.slice_range_from_center
    )
    print(f"train: {len(dataloaders[0].dataset)} samples")
    print(f"valid: {len(dataloaders[1].dataset)} samples")
    print(f"test: {len(dataloaders[2].dataset)} samples")
    print(f"unknown: {len(dataloaders[3].dataset)} samples")

    if "resnet" in args.backbone:
        cvae = ConditionalVAE(num_classes=3, latent_dim=256, base_channels=4,
                              backbone=args.backbone, weights="DEFAULT",
                              bound_z_by=args.bound_z_by)
    elif args.backbone.isdigit():
        args.backbone = f"cnn{int(args.backbone)}"
        cvae = ConditionalVAE(num_classes=3, latent_dim=256, base_channels=4,
                              downsample_factor=int(args.backbone),
                              bound_z_by=args.bound_z_by)
    cvae = cvae.to(args.device)

    ckpt_dir = os.path.join(args.ckpt_dir, f"{args.backbone}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}")
    train_cvae(cvae, train_loader=dataloaders[0], valid_loader=dataloaders[1],
               ckpt_dir=ckpt_dir,
               x_key="image",
               y_key="manufacturer_id",
               beta1=args.beta1,
               bootstrap=args.bootstrap,
               device=args.device,
               epochs=args.epochs,
               lr=args.lr,
               if_existing_ckpt=args.if_existing_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE for ADNI on Manufacturer")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/InvaRep/data/ADNI/",
                        help="Directory for ADNI data")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/InvaRep/checkpoints/adni",
                        help="Directory to save checkpoints")
    parser.add_argument("--backbone", type=str, default="4", help="Backbone architecture")
    parser.add_argument("--beta1", type=float, default=1.0, help="Beta1 parameter for CVAE loss")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--batch_per_epoch", type=int, default=100,
                        help="Number of batches per epoch")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Whether to bootstrap the dataset")
    parser.add_argument("--spatial_size", type=int, nargs=2, default=[224, 224],
                        help="Spatial size of the images")
    parser.add_argument("--slice_range_from_center", type=float, default=0.03,
                        help="Slice range from center for the images")
    parser.add_argument("--if_existing_ckpt", type=str, default="resume",
                        choices=["resume", "replace", "pass"],
                        help="What to do if an existing checkpoint is found")
    parser.add_argument("--bound_z_by", type=str, default=None,
                        choices=[None, "tanh", "standardization", "normalization"],
                        help="How to bound the latent space z")
    args = parser.parse_args()

    main(args)
