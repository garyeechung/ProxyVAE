import argparse
import os

import pandas as pd
import torch

from src.adni.models import VariationalPredictor
from src.adni.datasets import get_adni_dataloaders
from .model import train_infobottleneck


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
    print(f"Using device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"train: {len(dataloaders[0].dataset)} samples")
    print(f"valid: {len(dataloaders[1].dataset)} samples")
    print(f"test: {len(dataloaders[2].dataset)} samples")
    print(f"unknown: {len(dataloaders[3].dataset)} samples")

    group_model = VariationalPredictor(num_classes=3, backbone=args.backbone, weights="DEFAULT", is_posthoc=False)
    group_model = group_model.to(args.device)
    group_model = train_infobottleneck(model=group_model, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                                       ckpt_dir=os.path.join(args.ckpt_dir, args.backbone),
                                       x_key="image",
                                       y_key="manufacturer_id",
                                       beta=args.beta,
                                       bootstrap=args.bootstrap,
                                       device=args.device,
                                       epochs=args.epochs,
                                       lr=args.lr,
                                       if_existing_ckpt="resume")
    group_model = group_model.to("cpu")
    torch.cuda.empty_cache()

    group_model_best_path = os.path.join(args.ckpt_dir,
                                         args.backbone,
                                         "infobottleneck",
                                         f"beta_{args.beta:.1E}",
                                         "infobottleneck_manufacturer_id_best.pth")
    group_model_best_ckpt = torch.load(group_model_best_path, weights_only=False)
    group_model.load_state_dict(group_model_best_ckpt["model_state_dict"])
    for param in group_model.parameters():
        param.requires_grad = False

    class_model = VariationalPredictor(num_classes=9, encoder=group_model.encoder, is_posthoc=True)
    class_model = class_model.to(args.device)
    class_model = train_infobottleneck(model=class_model, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                                       ckpt_dir=os.path.join(args.ckpt_dir, args.backbone),
                                       x_key="image",
                                       y_key="model_type_id",
                                       beta=args.beta,
                                       bootstrap=args.bootstrap,
                                       device=args.device,
                                       epochs=args.epochs,
                                       lr=args.lr,
                                       if_existing_ckpt="resume")
    class_model = class_model.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train InfoBottleneck for ADNI on Manufacturer and Model Type")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/InvaRep/data/ADNI/",
                        help="Directory for ADNI data")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/InvaRep/checkpoints/adni",
                        help="Directory to save checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone architecture")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for InfoBottleneck loss")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--bootstrap", action="store_true", help="Use bootstrap sampling")
    parser.add_argument("--batch_per_epoch", type=int, default=1000,
                        help="Number of batches per epoch for training")
    parser.add_argument("--spatial_size", type=int, nargs=2, default=[224, 224],
                        help="Spatial size of the images")
    parser.add_argument("--slice_range_from_center", type=float, default=0.03,
                        help="Slice range from center for the images")
    parser.add_argument("--if_existing_ckpt", type=str, default="resume",
                        choices=["resume", "replace", "pass"],
                        help="What to do if an existing checkpoint is found")

    args = parser.parse_args()
    main(args)
