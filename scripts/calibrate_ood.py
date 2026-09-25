import argparse
import os
import sys
import json
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline import CattleBuffaloDataset, prepare_splits, _eval_transform
from src.model import BreedClassifier
from src.run_utils import find_latest_checkpoint, resolve_checkpoint
from src.config import OOD_ENERGY_TEMPERATURE, CHECKPOINT_DIR
from src.ood_detector import OODDetector


def main():
    parser = argparse.ArgumentParser(description="Calibrate OOD thresholds on the validation set")
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile of energy to use as threshold (e.g. 95)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate_ood] Using device: {device}")

    # Load model
    ckpt_path = resolve_checkpoint(args.checkpoint, args.backbone, CHECKPOINT_DIR)
    print(f"[calibrate_ood] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    binary_dim = 256
    for k, v in state_dict.items():
        if k == "binary_head.0.weight":
            binary_dim = v.shape[0]
            break

    model = BreedClassifier(backbone=args.backbone, binary_dim=binary_dim)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    # Load validation data
    prepare_splits()
    val_csv = os.path.join(PROJECT_ROOT, "data", "splits", "val.csv")
    if not os.path.exists(val_csv):
        print(f"[calibrate_ood] Error: {val_csv} not found.")
        return
        
    dataset = CattleBuffaloDataset(val_csv, transform=_eval_transform())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"[calibrate_ood] Validating on {len(dataset)} in-distribution images...")

    detector = OODDetector(temperature=OOD_ENERGY_TEMPERATURE)
    energies = []
    msps = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Calibrating"):
            images = images.to(device)
            out = model(images)
            
            # Compute energy
            energy_batch = detector.compute_energy(out["binary"], out["cattle"], out["buffalo"])
            msp_batch = detector.compute_msp(out["binary"])
            
            energies.extend(energy_batch.cpu().tolist())
            msps.extend(msp_batch.cpu().tolist())

    energies_t = torch.tensor(energies)
    msps_t = torch.tensor(msps)
    
    energy_threshold = torch.quantile(energies_t, args.percentile / 100.0).item()
    msp_threshold = torch.quantile(msps_t, (100.0 - args.percentile) / 100.0).item()
    
    print("\n[calibrate_ood] Calibration Results:")
    print(f"  Target percentile: {args.percentile}%")
    print(f"  Calibrated Energy Threshold: {energy_threshold:.4f}")
    print(f"  Calibrated MSP Threshold: {msp_threshold:.4f}")
    
    out_file = os.path.join(PROJECT_ROOT, "outputs", "ood_thresholds.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "energy_temperature": OOD_ENERGY_TEMPERATURE,
            "energy_threshold": energy_threshold,
            "msp_threshold": msp_threshold,
            "percentile": args.percentile,
            "calibrated_on": len(dataset)
        }, f, indent=2)
    print(f"[calibrate_ood] Saved thresholds to {out_file}")
    print("\nAction required: Please update src/config.py with these values if you want them to be the new defaults.")

if __name__ == "__main__":
    main()
