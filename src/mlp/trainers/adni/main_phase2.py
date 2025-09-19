import argparse
import os

import pandas as pd
import torch

from src.mlp.models import ConditionalVAE, ProxyVAE, VariationalPredictor, ProxyRep2InvaRep
from src.mlp.datasets import get_adni_dataloaders
from src.mlp.trainers.proxyvae import train_proxyvae, train_posthoc_predictor, train_proxy2invarep


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
    cvae_ckpt = os.path.join(ckpt_dir, "proxyvae", f"beta1_{args.beta1:.1E}", "cvae_best.pth")

    if not os.path.exists(cvae_ckpt):
        print(f"ConditionalVAE checkpoint not found at {cvae_ckpt}")
        return

    cvae_ckpt = torch.load(cvae_ckpt, weights_only=False)
    cvae = ConditionalVAE(num_classes=3, bound_z_by=args.bound_z_by)
    cvae.load_state_dict(cvae_ckpt["model_state_dict"])
    for param in cvae.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()

    # Second phase: Train the Proxy Variational Autoencoder (ProxyVAE)
    proxyvae = ProxyVAE(cvae=cvae)
    proxyvae = proxyvae.to(args.device)

    train_proxyvae(proxyvae, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                   ckpt_dir=ckpt_dir, x_key="image", dataset_name=f"adni_{args.modality}",
                   beta1=args.beta1, beta2=args.beta2, device=args.device, epochs=args.epochs,
                   lr=args.lr, if_existing_ckpt=args.if_existing_ckpt)
    proxyvae_model_best_path = os.path.join(args.ckpt_dir,
                                            f"{args.modality}{'_' + args.bound_z_by if args.bound_z_by is not None else ''}",
                                            "proxyvae",
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

    # Freeze the parameters of ProxyVAE, train z2 -> z1 predictor
    proxyvae = proxyvae.to("cpu")
    for param in proxyvae.parameters():
        param.requires_grad = False

    proxy2invarep = ProxyRep2InvaRep(proxyvae)
    proxy2invarep = proxy2invarep.to(args.device)
    print(f"Training ProxyRep2InvaRep with beta1={args.beta1}, beta2={args.beta2}")
    train_proxy2invarep(proxy2invarep, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                        ckpt_dir=ckpt_dir,
                        x_key="image",
                        dataset_name=f"adni_{args.modality}",
                        beta1=args.beta1,
                        beta2=args.beta2,
                        bound_z_by=args.bound_z_by,
                        device=args.device,
                        epochs=args.epochs,
                        lr=args.lr,
                        if_existing_ckpt="resume")
    proxy2invarep = proxy2invarep.to("cpu")
    torch.cuda.empty_cache()

    # Post-hoc predictor for manufacturer_id
    posthoc_coarse = VariationalPredictor(encoder=proxyvae.encoder2,
                                          num_classes=3, is_posthoc=True)
    posthoc_coarse = posthoc_coarse.to(args.device)
    train_posthoc_predictor(posthoc_coarse, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                            ckpt_dir=ckpt_dir,
                            x_key="image", y_key="manufacturer_id",
                            dataset_name=f"adni_{args.modality}",
                            beta1=args.beta1, beta2=args.beta2, device=args.device,
                            bound_z_by=args.bound_z_by, epochs=args.epochs,
                            lr=args.lr, if_existing_ckpt="resume")
    posthoc_coarse = posthoc_coarse.to("cpu")
    torch.cuda.empty_cache()

    # Post-hoc predictor for model_type_id
    posthoc_fine = VariationalPredictor(encoder=proxyvae.encoder2,
                                        num_classes=9, is_posthoc=True)
    posthoc_fine = posthoc_fine.to(args.device)
    train_posthoc_predictor(posthoc_fine, train_loader=dataloaders[0], valid_loader=dataloaders[1],
                            ckpt_dir=ckpt_dir,
                            x_key="image", y_key="model_type_id",
                            dataset_name=f"adni_{args.modality}",
                            beta1=args.beta1, beta2=args.beta2, device=args.device,
                            bound_z_by=args.bound_z_by, epochs=args.epochs,
                            lr=args.lr, if_existing_ckpt="resume")
    posthoc_fine = posthoc_fine.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProxyVAE and post-hocs for ADNI")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/data/ADNI/",
                        help="Directory for ADNI data")
    parser.add_argument("--modality", type=str, default="connectome", choices=["connectome"],
                        help="Modality to use for training (connectome)")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/chungk1/Repositories/ProxyVAE/checkpoints/adni",
                        help="Directory to save checkpoints")
    parser.add_argument("--beta1", type=float, default=1.0, help="Beta1 parameter for CVAE loss")
    parser.add_argument("--beta2", type=float, default=1.0, help="Beta2 parameter for ProxyVAE loss")
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
