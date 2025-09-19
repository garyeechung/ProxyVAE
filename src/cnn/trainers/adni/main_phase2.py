import argparse
import os

import pandas as pd
import torch

from src.cnn.models import ConditionalVAE, ProxyVAE, ProxyRep2InvaRep, VariationalPredictor
from src.cnn.datasets import get_adni_dataloaders
from src.cnn.trainers.invarep import train_proxyvae, train_proxy2invarep, train_posthoc_predictor


def main(args):
    df = pd.read_csv(os.path.join(args.data_dir, "adni_data.csv"))
    dataloaders = get_adni_dataloaders(
        df, data_dir=os.path.join(args.data_dir, "FA_rigid_MNI_1mm"),
        modality=args.modality, targets=["manufacturer_id", "model_type_id"],
        batch_size=args.batch_size, include_mappable_site_empty=True,
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

    ckpt_dir = os.path.join(args.ckpt_dir, args.modality, f"{args.backbone}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}")
    cvae_ckpt = os.path.join(ckpt_dir, "invarep", f"beta1_{args.beta1:.1E}", "cvae_best.pth")
    if not os.path.exists(cvae_ckpt):
        print(f"ConditionalVAE checkpoint not found at {cvae_ckpt}")
        return

    cvae_ckpt = torch.load(cvae_ckpt, weights_only=False)
    cvae = ConditionalVAE(num_classes=3, latent_dim=256, base_channels=4,
                          backbone=args.backbone, weights="DEFAULT",
                          bound_z_by=args.bound_z_by)
    cvae.load_state_dict(cvae_ckpt["model_state_dict"])
    for param in cvae.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()

    # Second phase: Train the Proxy Variational Autoencoder (ProxyVAE)
    proxyvae = ProxyVAE(cvae=cvae, latent_dim=256, base_channels=4,
                        backbone=args.backbone, weights="DEFAULT",
                        bound_z_by=args.bound_z_by)
    proxyvae = proxyvae.to(args.device)
    print(f"Training ProxyVAE with beta1={args.beta1}, beta2={args.beta2}")
    train_proxyvae(proxyvae, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                   ckpt_dir=ckpt_dir,
                   x_key="image",
                   dataset_name=f"adni_{args.modality}",
                   backbone=args.backbone,
                   beta1=args.beta1,
                   beta2=args.beta2,
                   bootstrap=args.bootstrap,
                   bound_z_by=args.bound_z_by,
                   device=args.device,
                   epochs=args.epochs * 4,
                   lr=args.lr * 10,
                   if_existing_ckpt="resume")
    proxyvae_model_best_path = os.path.join(args.ckpt_dir, args.modality,
                                            f"{args.backbone}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}",
                                            "invarep",
                                            f"beta1_{args.beta1:.1E}",
                                            f"beta2_{args.beta2:.1E}",
                                            "proxyvae_best.pth")
    proxyvae_model_best_ckpt = torch.load(proxyvae_model_best_path, weights_only=False)
    proxyvae.load_state_dict(proxyvae_model_best_ckpt["model_state_dict"])
    torch.cuda.empty_cache()

    # Freeze the parameters of ProxyVAE, train z2 -> z1 predictor
    proxyvae = proxyvae.to("cpu")
    for param in proxyvae.parameters():
        param.requires_grad = False

    proxy2invarep = ProxyRep2InvaRep(proxyvae, image_size=args.spatial_size, image_channels=3)
    proxy2invarep = proxy2invarep.to(args.device)
    print(f"Training ProxyRep2InvaRep with beta1={args.beta1}, beta2={args.beta2}")
    train_proxy2invarep(proxy2invarep, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                        ckpt_dir=ckpt_dir,
                        x_key="image",
                        dataset_name=f"adni_{args.modality}",
                        backbone=args.backbone,
                        beta1=args.beta1,
                        beta2=args.beta2,
                        bootstrap=args.bootstrap,
                        bound_z_by=args.bound_z_by,
                        device=args.device,
                        epochs=args.epochs,
                        lr=args.lr,
                        if_existing_ckpt="resume")
    proxy2invarep = proxy2invarep.to("cpu")
    torch.cuda.empty_cache()

    # Post-hoc predictor for manufacturer_id
    posthoc_group = VariationalPredictor(encoder=proxyvae.encoder2,
                                         num_classes=3, is_posthoc=True)
    posthoc_group = posthoc_group.to(args.device)
    print(f"Training post-hoc predictor for manufacturer_id with beta1={args.beta1}, beta2={args.beta2}")
    train_posthoc_predictor(posthoc_group, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                            ckpt_dir=ckpt_dir,
                            x_key="image", y_key="manufacturer_id",
                            dataset_name=f"adni_{args.modality}",
                            backbone=args.backbone,
                            beta1=args.beta1, beta2=args.beta2, device=args.device,
                            bound_z_by=args.bound_z_by,
                            bootstrap=args.bootstrap, epochs=args.epochs,
                            lr=args.lr, if_existing_ckpt="resume")
    posthoc_group = posthoc_group.to("cpu")
    torch.cuda.empty_cache()

    # Post-hoc predictor for model_type_id
    posthoc_class = VariationalPredictor(encoder=proxyvae.encoder2,
                                         num_classes=9, is_posthoc=True)
    posthoc_class = posthoc_class.to(args.device)
    print(f"Training post-hoc predictor for model_type_id with beta1={args.beta1}, beta2={args.beta2}")
    train_posthoc_predictor(posthoc_class, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                            ckpt_dir=ckpt_dir,
                            x_key="image", y_key="model_type_id",
                            dataset_name=f"adni_{args.modality}",
                            backbone=args.backbone,
                            beta1=args.beta1, beta2=args.beta2, device=args.device,
                            bound_z_by=args.bound_z_by,
                            bootstrap=args.bootstrap, epochs=args.epochs,
                            lr=args.lr, if_existing_ckpt="resume")
    posthoc_class = posthoc_class.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProxyVAE and post-hocs for ADNI")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/data/ADNI/",
                        help="Directory for ADNI data")
    parser.add_argument("--modality", type=str, default="fa", choices=["fa", "t1"],
                        help="Modality to use for training (fa or t1)")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/checkpoints/adni",
                        help="Directory to save checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone architecture")
    parser.add_argument("--beta1", type=float, default=1.0, help="Beta1 parameter for CVAE loss")
    parser.add_argument("--beta2", type=float, default=1.0, help="Beta2 parameter for ProxyVAE loss")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--batch_per_epoch", type=int, default=100,
                        help="Number of batches per epoch")
    parser.add_argument("--bootstrap", action="store_true", help="Whether to bootstrap the dataset")
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
