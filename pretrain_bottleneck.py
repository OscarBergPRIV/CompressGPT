#!/usr/bin/env python3
"""
Train AdaptiveBottleneck with MASKED loss (ignores padding)
+ Prints real vs padded lengths for every batch
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# ------------------- CONFIG & MODEL -------------------
try:
    from config import CFG_M
    from modeling_GPT_compress import AdaptiveBottleneck
except ImportError as e:
    print("Warning: Could not import config or modeling_GPT_compress")
    print("Make sure you're in the correct directory.")
    raise e


# ------------------- DATASET -------------------
class HiddenStateDataset(Dataset):
    def __init__(self, data_path: str, max_length: Optional[int] = None):
        self.data_path = Path(data_path)
        self.max_length = max_length

        self.file_paths = sorted(self.data_path.glob("batch_*.pt"))
        if not self.file_paths:
            raise ValueError(f"No batch_*.pt files in {data_path}")

        self.seq_per_file: List[int] = []
        self.cumulative: List[int] = [0]
        total_seqs = 0

        print(f"Scanning {len(self.file_paths)} files...")
        for fp in self.file_paths:
            batch = torch.load(fp, map_location="cpu")
            if not isinstance(batch, (list, tuple)):
                raise TypeError(f"{fp} must contain list/tuple of tensors")
            n = len(batch)
            self.seq_per_file.append(n)
            total_seqs += n
            self.cumulative.append(total_seqs)

        first_tensor = torch.load(self.file_paths[0], map_location="cpu")[0]
        self.hidden_dim = first_tensor.shape[-1]

        print(f"Total sequences      : {total_seqs:,}")
        print(f"Hidden dimension     : {self.hidden_dim}")
        print(f"Sequences per file   : {self.seq_per_file[:5]}{'...' if len(self.seq_per_file)>5 else ''}")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, idx: int):
        file_idx = next(i for i, cum in enumerate(self.cumulative) if cum > idx) - 1
        seq_idx = idx - self.cumulative[file_idx]
        batch = torch.load(self.file_paths[file_idx], map_location="cpu")
        seq = batch[seq_idx]

        if self.max_length is not None:
            if seq.shape[0] > self.max_length:
                seq = seq[:self.max_length]
            elif seq.shape[0] < self.max_length:
                pad = torch.zeros(self.max_length - seq.shape[0], self.hidden_dim)
                seq = torch.cat([seq, pad], dim=0)
        return seq


# ------------------- COLLATE + PRINT LENGTHS -------------------
# ------------------- COLLATE + PRINT LENGTHS -------------------
def collate_fn(batch):
    if len(batch) == 0:
        return torch.empty(0, 0, 768), torch.empty(0, dtype=torch.long)

    # === PRINT REAL LENGTHS (before padding) ===
    real_lengths = [s.shape[0] for s in batch]
    #print(f"\n[COLLATE] Real lengths (before pad): {real_lengths}")
    #print(f"          Min: {min(real_lengths)}, Max: {max(real_lengths)}, Mean: {np.mean(real_lengths):.1f}")

    max_len = max(real_lengths)
    hidden_dim = batch[0].shape[1]  # ← CORRECT
    dtype = batch[0].dtype

    padded = torch.zeros(len(batch), max_len, hidden_dim, dtype=dtype)
    lengths = []

    for i, seq in enumerate(batch):
        L = seq.shape[0]
        padded[i, :L] = seq
        lengths.append(L)

    # === PRINT AFTER PADDING ===
    #print(f"[COLLATE] Padded to: {max_len}")
    #print(f"          Padded tensor shape: {padded.shape}\n")

    return padded, torch.tensor(lengths, dtype=torch.long)


# ------------------- MASKED LOSS -------------------
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(self, pred, target, lengths):
        B, L, D = pred.shape
        device = pred.device

        # === CREATE MASK ===
        mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
        #print(mask)
        mask = mask.float()
        #print(f"[LOSS] Mask shape: {mask.shape} | Real tokens per seq: {lengths.tolist()}")

        # === APPLY MASK ===
        pred_masked = pred * mask.unsqueeze(-1)
        target_masked = target * mask.unsqueeze(-1)
        #print("pred masked.shape", pred_masked.shape)
        #print("target masked.shape", target_masked.shape)
        
        
        # === MSE (only real tokens) ===
        mse_loss = F.mse_loss(pred_masked, target_masked, reduction='sum') / mask.sum()

        # === COSINE (only real tokens) ===
        pred_flat = pred_masked[mask.bool()]
        target_flat = target_masked[mask.bool()]
        if pred_flat.numel() == 0:
            cosine_loss = torch.tensor(0.0, device=device)
        else:
            cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
            cosine_loss = 1 - cos_sim

        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
        return total_loss, mse_loss, cosine_loss


# ------------------- TRAINING LOOP -------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = total_mse = total_cos = 0.0
    #pbar = tqdm(loader, desc=f"Train {epoch}")

    for i, batch in enumerate(loader):
        seqs, lengths = batch
        seqs = seqs.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        recon = model(seqs)
        loss, mse, cos = criterion(recon, seqs, lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_cos += cos.item()

        if i == 0:  # Print only first batch
            print(f"[TRAIN] First batch loss: {loss.item():.4f} | MSE: {mse.item():.4f} | Cos: {cos.item():.4f}")

        #pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = len(loader)
    return total_loss/n, total_mse/n, total_cos/n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_mse = total_cos = 0.0
    with torch.no_grad():
        #pbar = tqdm(loader, desc="Val")
        for i, batch in enumerate(loader):
            seqs, lengths = batch
            seqs = seqs.to(device)
            lengths = lengths.to(device)

            recon = model(seqs)
            loss, mse, cos = criterion(recon, seqs, lengths)

            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()

            if i == 0:
                print(f"[VAL] First batch loss: {loss.item():.4f} | MSE: {mse.item():.4f} | Cos: {cos.item():.4f}")

    n = len(loader)
    return total_loss/n, total_mse/n, total_cos/n


# ------------------- CHECKPOINT -------------------
def save_checkpoint(model, optimizer, epoch, loss, path, args):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args),
    }, path)
    print(f"Checkpoint → {path}")


# ------------------- MAIN -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='collected_hidden_states/real_hidden_data_3_16')
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='checkpoints_bottleneck')
    parser.add_argument('--model_name', type=str, default='adaptive_bottleneck')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_config', action='store_true')
    args = parser.parse_args()

    cfg_m = CFG_M()
    print("Chosen ratio: ", cfg_m.bl_ratio)

    # SLURM: disable workers
    num_workers = args.num_workers
    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM detected → num_workers=0")
        num_workers = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #out_dir = Path(args.output_dir) / f"{args.model_name}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.output_dir) / f"_r_{cfg_m.bl_ratio}_pos_{cfg_m.bl_layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = HiddenStateDataset(args.data_path, max_length=args.max_length)
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    #if args.use_config:
    #    cfg = CFG_M()
    #    model = AdaptiveBottleneck(cfg)
    #else:
    
    if cfg_m.BL_type == "linear":
        model = AdaptiveBottleneck(ratio=cfg_m.bl_ratio)
    model.to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = CombinedLoss(args.mse_weight, args.cosine_weight)

    best_val = float('inf')
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_mse, train_cos = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_mse, val_cos = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, out_dir / f"epoch_{epoch}.pt", args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, out_dir / "best.pt", args)
            print("New best!")

    save_checkpoint(model, optimizer, args.num_epochs, val_loss, out_dir / "final.pt", args)
    print(f"Training done! Best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
