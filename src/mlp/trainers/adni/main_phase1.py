import argparse
import os

import pandas as pd
import torch

from src.mlp.models import ConditionalVAE
from src.mlp.datasets import get_adni_dataloaders
from src.mlp.trainers.methods.proxyvae import train_cvae
from src.mlp.datasets.connectome.utils import ADNI_MERGE_GROUP, ADNI_COARSE_MAPPING, ADNI_FINE_MAPPING, vis_x_recon_comparison


TSNE_CONFIG = {
    "merge_group": ADNI_MERGE_GROUP,
    "coarse_mapping": ADNI_COARSE_MAPPING,
    "fine_mapping": ADNI_FINE_MAPPING
}


def main(args):
    df = pd.read_csv(os.path.join(args.data_dir, "adni_data_connectome.csv"))
    dataloaders = get_adni_dataloaders(
        df, data_dir=os.path.join(args.data_dir, "FA_rigid_MNI_1mm"),
        modality=args.modality, targets=["manufacturer_id", "model_type_id"],
        num_classes=[3, 9], one_hot=True,
        batch_size=args.batch_size, include_mappable_site_empty=False,
        num_workers=0, batch_per_epoch=args.batch_per_epoch,
    )
    print(f"train: {len(dataloaders[0].dataset)} samples")
    print(f"valid: {len(dataloaders[1].dataset)} samples")
    print(f"test: {len(dataloaders[2].dataset)} samples")

    cvae = ConditionalVAE(num_classes=3, bound_z_by=args.bound_z_by)
    cvae = cvae.to(args.device)
    sub_path = f"{args.modality}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}"
    ckpt_dir = os.path.join(args.ckpt_dir, sub_path)
    train_cvae(cvae, train_loader=dataloaders[0], valid_loader=dataloaders[1],
               ckpt_dir=ckpt_dir,
               x_key="image",
               y_key="manufacturer_id",
               dataset_name=f"adni_{args.modality}",
               beta1=args.beta1,
               bound_z_by=args.bound_z_by,
               device=args.device,
               epochs=args.epochs,
               lr=args.lr,
               if_existing_ckpt=args.if_existing_ckpt,
               tsne_config=TSNE_CONFIG,
               comparison_fn=vis_x_recon_comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE for ADNI on Manufacturer")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/data/ADNI/",
                        help="Directory for ADNI data")
    parser.add_argument("--modality", type=str, default="connectome", choices=["connectome"],
                        help="Modality to use for training (connectome)")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/checkpoints/adni",
                        help="Directory to save checkpoints")
    parser.add_argument("--beta1", type=float, default=1.0, help="Beta1 parameter for CVAE loss")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--batch_per_epoch", type=int, default=100,
                        help="Number of batches per epoch")
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
