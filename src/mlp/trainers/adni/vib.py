import argparse
import os

import pandas as pd
import torch

from src.mlp.models import VariationalPredictor
from src.mlp.datasets import get_adni_dataloaders
from src.mlp.trainers.methods.vib import train_vib
from src.mlp.trainers.methods.proxyvae import train_posthoc_predictor
from src.mlp.datasets.connectome.utils import ADNI_MERGE_GROUP, ADNI_COARSE_MAPPING, ADNI_FINE_MAPPING


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

    ckpt_dir = os.path.join(args.ckpt_dir, f"{args.modality}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}")

    vib_model = VariationalPredictor(num_classes=3, is_posthoc=False, bound_z_by=args.bound_z_by)
    vib_model = vib_model.to(args.device)

    train_vib(vib_model, train_loader=dataloaders[0], valid_loader=dataloaders[1],
              ckpt_dir=ckpt_dir, x_key="image", y_key="manufacturer_id",
              dataset_name=f"adni_{args.modality}",
              beta=args.beta, device=args.device,
              bound_z_by=args.bound_z_by,
              tsne_config=TSNE_CONFIG, epochs=args.epochs,
              lr=args.lr, if_existing_ckpt=args.if_existing_ckpt)
    vib_model_best_path = os.path.join(args.ckpt_dir,
                                       f"{args.modality}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}",
                                       "vib",
                                       f"beta_{args.beta:.1E}",
                                       "vib_best.pth")
    vib_model_best_ckpt = torch.load(vib_model_best_path, weights_only=False)
    vib_model.load_state_dict(vib_model_best_ckpt["model_state_dict"])
    for param in vib_model.parameters():
        param.requires_grad = False
    vib_model = vib_model.to("cpu")
    torch.cuda.empty_cache()

    posthoc_model = VariationalPredictor(num_classes=9, is_posthoc=True, encoder=vib_model.encoder, bound_z_by=args.bound_z_by)
    posthoc_model = posthoc_model.to(args.device)
    train_posthoc_predictor(posthoc_model, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                            ckpt_dir=ckpt_dir, x_key="image", y_key="model_type_id",
                            dataset_name=f"adni_{args.modality}",
                            device=args.device, epochs=args.epochs,
                            lr=args.lr, if_existing_ckpt=args.if_existing_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VIB for ADNI on Manufacturer")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/data/ADNI/",
                        help="Directory for ADNI data")
    parser.add_argument("--modality", type=str, default="connectome", choices=["connectome"],
                        help="Modality to use for training (connectome)")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/checkpoints/adni",
                        help="Directory for saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--batch_per_epoch", type=int, default=100,
                        help="Number of batches per epoch (for epoch definition)")
    parser.add_argument("--bound_z_by", type=str, default=None, choices=[None, "tanh", "sigmoid"],
                        help="Bound latent space z by a function")
    parser.add_argument("--beta", type=float, default=1e-3, help="Beta value for VIB loss")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for optimizer")
    parser.add_argument("--if_existing_ckpt", type=str, default="resume", choices=["resume", "reset", "pass"],
                        help="Action if existing checkpoint is found: resume, reset, or pass")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()
    print(f"Using device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    main(args)
