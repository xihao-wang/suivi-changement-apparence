#!/usr/bin/env python3
"""Sequence-level TBPTT fine-tuning for DMAT (Dynamic Memory Aggregation Token).

Fine-tunes a pretrained TemporalAttentionScorer so the per-track DMAT query
learns to adapt to appearance changes across K consecutive frames.

Typical usage:
  # 1. Build sequence data (run build_temporal_sequences_from_gt.py first)
  # 2. Fine-tune on top of a pretrained base checkpoint:
  python tools/train_temporal_model_seq.py \
      --train_seq_npz data/temporal_sequences/new-1_seq.npz ... \
      --val_seq_npz   data/temporal_sequences/YT-03_seq.npz \
      --pretrained_ckpt data/checkpoints/infonce_track_fused/best.pt \
      --save_dir data/checkpoints/dmat_seq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_sort.temporal_model import TemporalAttentionScorer


# ── Dataset ──────────────────────────────────────────────────────────────────

class TemporalSequenceDataset(Dataset):
    """K-frame sequences built by build_temporal_sequences_from_gt.py."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        required = {"pos_det", "hist_short", "hist_long", "hist_slen", "hist_llen", "neg_det"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{npz_path} missing fields: {sorted(missing)}")
        self.pos_det    = data["pos_det"].astype(np.float32)    # (N, K, D)
        self.hist_short = data["hist_short"].astype(np.float32) # (N, K, Ls, D)
        self.hist_long  = data["hist_long"].astype(np.float32)  # (N, K, Ll, D)
        self.hist_slen  = data["hist_slen"].astype(np.int64)    # (N, K)
        self.hist_llen  = data["hist_llen"].astype(np.int64)    # (N, K)
        self.neg_det    = data["neg_det"].astype(np.float32)    # (N, K, M, D)

    def __len__(self):
        return len(self.pos_det)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.pos_det[idx]),
            torch.from_numpy(self.hist_short[idx]),
            torch.from_numpy(self.hist_long[idx]),
            torch.from_numpy(self.hist_slen[idx]),
            torch.from_numpy(self.hist_llen[idx]),
            torch.from_numpy(self.neg_det[idx]),
        )


class MultiTemporalSequenceDataset(Dataset):
    """Concatenation of multiple sequence npz files.

    Two optional mechanisms to make training harder:

    1. Global negative bank (use_global_neg=True):
       Each sample's negatives are drawn from other sequences' positive
       detection features, giving genuinely diverse competitors instead of
       always comparing against the same co-visible identity.

    2. Oversampling (oversample_paths + oversample_factor):
       Specified npz files are repeated `oversample_factor` times so their
       windows (e.g. appearance-change windows in new-4) appear more often.
    """

    def __init__(
        self,
        npz_paths: list[str],
        oversample_paths: list[str] | None = None,
        oversample_factor: int = 5,
        use_global_neg: bool = False,
        num_global_neg: int = 16,
    ):
        if not npz_paths:
            raise ValueError("npz_paths must not be empty")

        oversample_set = {str(Path(p).resolve()) for p in (oversample_paths or [])}

        parts_data   = []
        source_ids   = []   # index into npz_paths for each row

        for src_idx, npz_path in enumerate(npz_paths):
            part    = TemporalSequenceDataset(npz_path)
            factor  = oversample_factor if str(Path(npz_path).resolve()) in oversample_set else 1
            n       = len(part)
            for _ in range(factor):
                parts_data.append(part)
                source_ids.append(np.full(n, src_idx, dtype=np.int32))

        self.pos_det    = np.concatenate([p.pos_det    for p in parts_data], axis=0)
        self.hist_short = np.concatenate([p.hist_short for p in parts_data], axis=0)
        self.hist_long  = np.concatenate([p.hist_long  for p in parts_data], axis=0)
        self.hist_slen  = np.concatenate([p.hist_slen  for p in parts_data], axis=0)
        self.hist_llen  = np.concatenate([p.hist_llen  for p in parts_data], axis=0)
        self.neg_det    = np.concatenate([p.neg_det    for p in parts_data], axis=0)
        self.source_ids = np.concatenate(source_ids, axis=0)

        # ── Global negative bank ──────────────────────────────────────────
        self.use_global_neg = use_global_neg
        self.num_global_neg = num_global_neg
        if use_global_neg:
            N, K, D = self.pos_det.shape
            # Flatten all positive features; remember which source each came from
            self._bank_feats  = self.pos_det.reshape(N * K, D)           # (N*K, D)
            self._bank_source = np.repeat(self.source_ids, K)             # (N*K,)

    def __len__(self):
        return len(self.pos_det)

    def _sample_global_negs(self, idx: int) -> np.ndarray:
        """Sample num_global_neg features from other-source entries in the bank."""
        K, D = self.pos_det.shape[1], self.pos_det.shape[2]
        M    = self.num_global_neg
        src  = int(self.source_ids[idx])

        mask       = self._bank_source != src
        candidates = np.where(mask)[0]

        if len(candidates) == 0:
            # Single-source fallback: repeat stored negatives
            stored = self.neg_det[idx]                          # (K, M_old, D)
            M_old  = stored.shape[1]
            idxs   = np.random.choice(M_old, size=(K, M), replace=(M > M_old))
            return stored[np.arange(K)[:, None], idxs]

        total = K * M
        idxs  = np.random.choice(candidates, size=total, replace=(len(candidates) < total))
        return self._bank_feats[idxs].reshape(K, M, D)

    def __getitem__(self, idx):
        neg = self._sample_global_negs(idx) if self.use_global_neg else self.neg_det[idx]
        return (
            torch.from_numpy(self.pos_det[idx]),
            torch.from_numpy(self.hist_short[idx]),
            torch.from_numpy(self.hist_long[idx]),
            torch.from_numpy(self.hist_slen[idx]),
            torch.from_numpy(self.hist_llen[idx]),
            torch.from_numpy(neg.astype(np.float32)),
        )


# ── Core TBPTT logic ─────────────────────────────────────────────────────────

def _score_batch(model, det, hist_s, hist_l, slen, llen, dmat):
    """Score one detection against one track state; return (logit, updated_dmat).

    updated_dmat is None when dmat is None (model returns a scalar then).
    """
    result = model(
        det, hist_s,
        long_hist_feat=hist_l,
        short_hist_len=slen,
        long_hist_len=llen,
        dmat=dmat,
        return_attention=False,
    )
    if isinstance(result, tuple):
        return result[0], result[1]   # score (B,), updated_dmat (B, 1, H)
    return result, None


def _add_feat_noise(feat: torch.Tensor, std: float) -> torch.Tensor:
    """Add isotropic Gaussian noise then re-normalise to unit sphere."""
    return F.normalize(feat + torch.randn_like(feat) * std, dim=-1)


def tbptt_forward(model, pos_det, hist_short, hist_long, hist_slen, hist_llen, neg_det, device,
                  feat_noise_std: float = 0.0):
    """One TBPTT sequence forward pass.

    At each step k:
      1. Score positive detection with current DMAT  →  pos_logit, updated_dmat
      2. Score M negative detections with same DMAT  →  neg_logits
      3. InfoNCE loss: positive must outscore all negatives
      4. TBPTT detach: pass updated_dmat.detach() to step k+1

    feat_noise_std > 0 adds Gaussian noise to the current-frame detection
    features only (not to history), forcing the model to rely on temporal
    context rather than raw static similarity.

    Returns mean InfoNCE loss over K steps.
    """
    B, K, D  = pos_det.shape
    M        = neg_det.shape[2]
    Ls       = hist_short.shape[2]
    Ll       = hist_long.shape[2]

    # Start each sequence with the model's learned initial query.
    dmat = model.long_query_token.expand(B, -1, -1).detach().clone()  # (B, 1, H)

    total_loss = pos_det.new_tensor(0.0)

    for k in range(K):
        hs_k   = hist_short[:, k]               # (B, Ls, D)
        hl_k   = hist_long[:, k]                # (B, Ll, D)
        slen_k = hist_slen[:, k]                # (B,)
        llen_k = hist_llen[:, k]                # (B,)

        # ── positive ──────────────────────────────────────────────────────
        pos_k = pos_det[:, k]
        if feat_noise_std > 0.0:
            pos_k = _add_feat_noise(pos_k, feat_noise_std)

        pos_logit, updated_dmat = _score_batch(
            model, pos_k, hs_k, hl_k, slen_k, llen_k, dmat
        )
        # pos_logit: (B,)   updated_dmat: (B, 1, H)

        # ── negatives: same track history, M different detections ─────────
        neg_flat  = neg_det[:, k].reshape(B * M, D)
        if feat_noise_std > 0.0:
            neg_flat = _add_feat_noise(neg_flat, feat_noise_std)
        hs_rep    = hs_k.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, Ls, D)
        hl_rep    = hl_k.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, Ll, D)
        slen_rep  = slen_k.unsqueeze(1).expand(-1, M).reshape(B * M)
        llen_rep  = llen_k.unsqueeze(1).expand(-1, M).reshape(B * M)
        dmat_rep  = dmat.repeat_interleave(M, dim=0)  # (B*M, 1, H)

        neg_logit_flat, _ = _score_batch(
            model, neg_flat, hs_rep, hl_rep, slen_rep, llen_rep, dmat_rep
        )
        neg_logits = neg_logit_flat.reshape(B, M)    # (B, M)

        # ── InfoNCE: positive at index 0 ──────────────────────────────────
        all_logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)  # (B, M+1)
        targets    = torch.zeros(B, dtype=torch.long, device=device)
        total_loss = total_loss + F.cross_entropy(all_logits, targets)

        # ── TBPTT detach: DMAT value flows forward, gradient does not ─────
        dmat = updated_dmat.detach()

    return total_loss / K


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_seq(model, loader, device):
    model.eval()
    total_loss  = 0.0
    total_steps = 0
    total_pos_wins = 0  # steps where pos_logit > all neg_logits

    with torch.no_grad():
        for pos_det, hist_short, hist_long, hist_slen, hist_llen, neg_det in loader:
            pos_det    = pos_det.to(device)
            hist_short = hist_short.to(device)
            hist_long  = hist_long.to(device)
            hist_slen  = hist_slen.to(device)
            hist_llen  = hist_llen.to(device)
            neg_det    = neg_det.to(device)

            B, K, D = pos_det.shape
            M  = neg_det.shape[2]
            Ls = hist_short.shape[2]
            Ll = hist_long.shape[2]

            dmat = model.long_query_token.expand(B, -1, -1).detach().clone()

            for k in range(K):
                hs_k   = hist_short[:, k]
                hl_k   = hist_long[:, k]
                slen_k = hist_slen[:, k]
                llen_k = hist_llen[:, k]

                pos_logit, updated_dmat = _score_batch(
                    model, pos_det[:, k], hs_k, hl_k, slen_k, llen_k, dmat
                )

                neg_flat = neg_det[:, k].reshape(B * M, D)
                hs_rep   = hs_k.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, Ls, D)
                hl_rep   = hl_k.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, Ll, D)
                slen_rep = slen_k.unsqueeze(1).expand(-1, M).reshape(B * M)
                llen_rep = llen_k.unsqueeze(1).expand(-1, M).reshape(B * M)
                dmat_rep = dmat.repeat_interleave(M, dim=0)

                neg_logit_flat, _ = _score_batch(
                    model, neg_flat, hs_rep, hl_rep, slen_rep, llen_rep, dmat_rep
                )
                neg_logits = neg_logit_flat.reshape(B, M)

                all_logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
                targets    = torch.zeros(B, dtype=torch.long, device=device)
                loss_k     = F.cross_entropy(all_logits, targets)

                total_loss     += float(loss_k.item()) * B
                total_steps    += B
                total_pos_wins += int((pos_logit > neg_logits.max(dim=1).values).sum().item())

                dmat = updated_dmat

    avg_loss = total_loss / max(total_steps, 1)
    acc      = total_pos_wins / max(total_steps, 1)
    return avg_loss, acc


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TBPTT sequence-level fine-tuning of TemporalAttentionScorer for DMAT."
    )
    parser.add_argument("--train_seq_npz", nargs="+", required=True,
                        help="One or more sequence npz files for training.")
    parser.add_argument("--val_seq_npz", nargs="+", required=True,
                        help="One or more sequence npz files for validation.")
    parser.add_argument("--pretrained_ckpt", required=True,
                        help="Base checkpoint from train_temporal_model.py (must have use_long_memory=True).")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Fine-tuning LR (lower than base training).")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clip (0 = disabled).")
    # ── Hard-negative options ─────────────────────────────────────────────
    parser.add_argument("--global_neg", action="store_true",
                        help="Replace stored negatives with features sampled from other "
                             "sequences (cross-sequence negative bank). Recommended: gives "
                             "genuinely diverse competitors instead of always the same person.")
    parser.add_argument("--num_global_neg", type=int, default=16,
                        help="Number of negatives per step when --global_neg is set.")
    parser.add_argument("--oversample_npz", nargs="*", default=None,
                        help="Paths from --train_seq_npz to repeat (oversample) in the dataset. "
                             "Use to up-weight sequences containing hard appearance changes.")
    parser.add_argument("--oversample_factor", type=int, default=5,
                        help="How many times to repeat oversampled npz files.")
    # ── Feature-noise augmentation ────────────────────────────────────────
    parser.add_argument("--feat_noise_std", type=float, default=0.0,
                        help="Std of isotropic Gaussian noise added to current-frame detection "
                             "features (pos and neg) at each TBPTT step during training only. "
                             "Noise is applied before L2 re-normalisation. "
                             "Forces the model to rely on temporal (DMAT) context rather than "
                             "raw static similarity. Try 0.05–0.15; 0 = disabled (default).")
    # ─────────────────────────────────────────────────────────────────────
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load pretrained model ─────────────────────────────────────────────
    ckpt = torch.load(args.pretrained_ckpt, map_location=args.device, weights_only=False)
    if not ckpt.get("use_long_memory", True):
        raise ValueError("Pretrained model has use_long_memory=False; DMAT needs long memory.")

    model = TemporalAttentionScorer(
        feature_dim=ckpt["feature_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_heads=ckpt.get("num_heads", 4),
        history_len=ckpt["history_len"],
        long_history_len=ckpt["long_history_len"],
        use_long_memory=True,
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    print(
        f"Loaded pretrained model from {args.pretrained_ckpt}  "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f}  "
        f"feature_dim={ckpt['feature_dim']}  hidden_dim={ckpt['hidden_dim']}"
    )

    # ── Datasets ──────────────────────────────────────────────────────────
    train_set = MultiTemporalSequenceDataset(
        args.train_seq_npz,
        oversample_paths=args.oversample_npz,
        oversample_factor=args.oversample_factor,
        use_global_neg=args.global_neg,
        num_global_neg=args.num_global_neg,
    )
    val_set = MultiTemporalSequenceDataset(args.val_seq_npz)

    K = int(train_set.pos_det.shape[1])
    M = args.num_global_neg if args.global_neg else int(train_set.neg_det.shape[2])
    D = int(train_set.pos_det.shape[2])

    oversample_info = ""
    if args.oversample_npz:
        names = [Path(p).stem for p in args.oversample_npz]
        oversample_info = f" | oversample x{args.oversample_factor}: {names}"
    neg_info = f"global ({M})" if args.global_neg else f"stored ({M})"
    noise_info = f" | feat_noise_std={args.feat_noise_std}" if args.feat_noise_std > 0 else ""
    print(
        f"Train: {len(train_set)} sequences | Val: {len(val_set)} sequences | "
        f"K={K} steps | negatives: {neg_info} | feat_dim={D}{oversample_info}{noise_info}"
    )

    # Verify feature dim matches model
    if D != ckpt["feature_dim"]:
        raise ValueError(
            f"Sequence feature dim ({D}) doesn't match model ({ckpt['feature_dim']}). "
            "Re-build sequences with the same detection files used for base training."
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch    = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for pos_det, hist_short, hist_long, hist_slen, hist_llen, neg_det in train_loader:
            pos_det    = pos_det.to(args.device)
            hist_short = hist_short.to(args.device)
            hist_long  = hist_long.to(args.device)
            hist_slen  = hist_slen.to(args.device)
            hist_llen  = hist_llen.to(args.device)
            neg_det    = neg_det.to(args.device)

            optimizer.zero_grad()
            loss = tbptt_forward(
                model, pos_det, hist_short, hist_long,
                hist_slen, hist_llen, neg_det, args.device,
                feat_noise_std=args.feat_noise_std,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches    += 1

        train_loss          = running_loss / max(n_batches, 1)
        val_loss, val_acc   = evaluate_seq(model, val_loader, args.device)

        ckpt_out = {
            "epoch":            epoch,
            "state_dict":       model.state_dict(),
            "feature_dim":      ckpt["feature_dim"],
            "hidden_dim":       ckpt["hidden_dim"],
            "num_heads":        ckpt.get("num_heads", 4),
            "history_len":      ckpt["history_len"],
            "long_history_len": ckpt["long_history_len"],
            "use_long_memory":  True,
            "train_loss":       train_loss,
            "val_loss":         val_loss,
            "val_acc":          val_acc,
            "seq_len":          K,
            "num_negatives":    M,
            "feat_noise_std":   args.feat_noise_std,
        }
        torch.save(ckpt_out, save_dir / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(ckpt_out, save_dir / "best.pt")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
            + (" ← best" if epoch == best_epoch else "")
        )

    print(f"Done. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
