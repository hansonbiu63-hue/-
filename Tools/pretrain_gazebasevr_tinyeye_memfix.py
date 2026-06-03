#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory-friendly GazeBaseVR pretraining for TinyEyeTransformer.

This script is designed for pretraining the temporal encoder on GazeBaseVR CSV
files, then saving a checkpoint that can be loaded by the Unity finetune script.

Expected GazeBaseVR raw CSV columns:
    n, x, y, lx, ly, rx, ry, xT, yT, zT, clx, cly, clz, crx, cry, crz

Typical usage:
    python pretrain_gazebasevr_tinyeye_memfix.py ^
      --gazebasevr_dir "F:\\GazeBaseVR\\data" ^
      --out_dir "F:\\TestData\\OUT_DIR_PRETRAIN" ^
      --step_frames 30 ^
      --max_windows_per_file 120 ^
      --max_total_windows 30000 ^
      --storage_dtype float16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


AOI_COUNT = 10
FEATURE_NAMES = [
    "screen_x_norm", "screen_y_norm", "delta_x", "delta_y", "gaze_speed",
    "gaze_dir_x", "gaze_dir_y", "gaze_dir_z", "gaze_dir_speed",
    "left_openness", "right_openness", "openness_mean", "openness_delta",
    "pupil_diameter", "pupil_valid", "tracking_valid", "hit_ui_plane",
    "aoi_onehot_0", "aoi_onehot_1", "aoi_onehot_2", "aoi_onehot_3",
    "aoi_onehot_4", "aoi_onehot_5", "aoi_onehot_6", "aoi_onehot_7",
    "aoi_onehot_8", "aoi_onehot_9", "pupil_mean", "pupil_std",
    "speed_p50", "speed_p90", "valid_ratio", "aoi_transition_rate",
    "aoi_entropy", "window_openness_mean", "window_openness_std",
    "blink_ratio", "aoi_dwell_0", "aoi_dwell_1", "aoi_dwell_2",
    "aoi_dwell_3", "aoi_dwell_4", "aoi_dwell_5", "aoi_dwell_6",
    "aoi_dwell_7", "aoi_dwell_8", "aoi_dwell_9",
]
FEATURE_DIM = len(FEATURE_NAMES)

# Only continuous columns are normalized. Binary and one-hot columns stay as-is.
CONTINUOUS_IDXS = list(range(0, 14)) + list(range(27, 47))
BINARY_IDXS = [14, 15, 16]
ONEHOT_IDXS = list(range(17, 27))
WINDOW_IDXS = list(range(27, 47))


@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray
    group: str
    source: str


class EyeDataset(Dataset):
    def __init__(self, samples: List[Sample], mean: np.ndarray, std: np.ndarray):
        self.samples = list(samples)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = s.x.astype(np.float32, copy=True)
        denom = np.where(np.abs(self.std) < 1e-6, 1.0, self.std)
        x[:, CONTINUOUS_IDXS] = (
            x[:, CONTINUOUS_IDXS] - self.mean[CONTINUOUS_IDXS]
        ) / denom[CONTINUOUS_IDXS]
        return torch.from_numpy(x), torch.from_numpy(s.y.astype(np.float32))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyEyeTransformer(nn.Module):
    """Same backbone keys as the finetune script.

    In pretrain mode, the final head is a task classifier over GazeBaseVR task
    types. During finetune, only compatible tensors such as proj/encoder/gate
    should be loaded.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = "pretrain"
        self.num_classes = num_classes

        self.proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Feature 15 is tracking_valid.
        invalid = x[:, :, 15] <= 0.5
        all_invalid = invalid.all(dim=1)
        if all_invalid.any():
            invalid = invalid.clone()
            invalid[all_invalid, 0] = False
        return invalid

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.padding_mask(x)
        z = self.encoder(self.pos(self.proj(x)), src_key_padding_mask=mask)
        return z, mask

    def weighted_gap(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float().unsqueeze(-1)
        w = self.gate(z) * valid
        return (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mask = self.encode(x)
        pooled = self.weighted_gap(z, mask)
        return self.head(pooled)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gazebasevr_dir", default=r"F:\GazeBaseVR\data")
    p.add_argument("--out_dir", default=r"F:\TestData\OUT_DIR_PRETRAIN")
    p.add_argument("--target_hz", type=int, default=50)
    p.add_argument("--source_hz", type=int, default=250)
    p.add_argument("--window_frames", type=int, default=150)
    p.add_argument("--step_frames", type=int, default=30)
    p.add_argument("--min_valid_frames", type=int, default=20)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_windows_per_file", type=int, default=120)
    p.add_argument("--max_total_windows", type=int, default=30000)
    p.add_argument("--storage_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def to_float(series, default=np.nan) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(np.float32)


def normalize_axis(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    ok = valid & np.isfinite(values)
    if not ok.any():
        return out

    vv = values[ok]
    # Already normalized.
    if float(np.nanmin(vv)) >= 0 and float(np.nanmax(vv)) <= 1:
        out[ok] = np.clip(vv, 0, 1)
        return out

    lo, hi = np.percentile(vv, [1, 99])
    if abs(float(hi - lo)) < 1e-6:
        lo, hi = float(np.nanmin(vv)), float(np.nanmax(vv))
    if abs(float(hi - lo)) >= 1e-6:
        out[ok] = np.clip((vv - lo) / (hi - lo), 0, 1)
    return out


def append_window_features(
    x: np.ndarray,
    pupil: np.ndarray,
    pupil_valid: np.ndarray,
    tracking: np.ndarray,
    aoi_id: np.ndarray,
    left_open: np.ndarray,
    right_open: np.ndarray,
    speed: np.ndarray,
):
    valid_pupil = pupil[pupil_valid]
    x[:, 27] = float(np.mean(valid_pupil)) if len(valid_pupil) else 0.0
    x[:, 28] = float(np.std(valid_pupil)) if len(valid_pupil) else 0.0

    finite_speed = speed[np.isfinite(speed)]
    x[:, 29] = float(np.percentile(finite_speed, 50)) if len(finite_speed) else 0.0
    x[:, 30] = float(np.percentile(finite_speed, 90)) if len(finite_speed) else 0.0
    x[:, 31] = float(np.mean(tracking > 0.5)) if len(tracking) else 0.0

    valid_aoi = aoi_id[(aoi_id >= 0) & (aoi_id < AOI_COUNT)]
    dwell = np.zeros(AOI_COUNT, dtype=np.float32)
    if len(valid_aoi):
        for aid in valid_aoi:
            dwell[int(aid)] += 1.0
        dwell /= float(len(valid_aoi))

    x[:, 32] = float(np.sum(valid_aoi[1:] != valid_aoi[:-1])) / max(1, len(valid_aoi) - 1) if len(valid_aoi) > 1 else 0.0
    nonzero = dwell[dwell > 0]
    x[:, 33] = float(-(nonzero * np.log(nonzero + 1e-8)).sum()) if len(nonzero) else 0.0

    openness = 0.5 * (left_open + right_open)
    open_valid = openness[np.isfinite(openness)]
    x[:, 34] = float(np.mean(open_valid)) if len(open_valid) else 0.0
    x[:, 35] = float(np.std(open_valid)) if len(open_valid) else 0.0
    x[:, 36] = float(np.mean(openness <= 0.05)) if len(openness) else 0.0
    x[:, 37:47] = dwell


def build_features(
    sx_raw: np.ndarray,
    sy_raw: np.ndarray,
    ts_ms: np.ndarray,
    tracking: np.ndarray,
    hit: np.ndarray,
    pupil: np.ndarray | None = None,
    pupil_valid: np.ndarray | None = None,
    gaze_dir: np.ndarray | None = None,
    left_open: np.ndarray | None = None,
    right_open: np.ndarray | None = None,
    aoi_id: np.ndarray | None = None,
) -> np.ndarray:
    n = len(sx_raw)
    x = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    valid = (tracking > 0.5) & (hit > 0.5) & np.isfinite(sx_raw) & np.isfinite(sy_raw)
    sx = normalize_axis(sx_raw.astype(np.float32), valid)
    sy = normalize_axis(sy_raw.astype(np.float32), valid)

    dt = np.diff(ts_ms.astype(np.float64), prepend=ts_ms[:1]) / 1000.0
    dt = np.where(dt <= 1e-4, 1.0 / 50.0, dt)

    dx = np.diff(sx, prepend=sx[:1])
    dy = np.diff(sy, prepend=sy[:1])
    speed = np.sqrt(dx * dx + dy * dy) / dt

    gaze_dir = np.zeros((n, 3), np.float32) if gaze_dir is None else gaze_dir.astype(np.float32)
    dir_speed = np.linalg.norm(np.diff(gaze_dir, axis=0, prepend=gaze_dir[:1]), axis=1) / dt

    pupil = np.zeros(n, np.float32) if pupil is None else pupil.astype(np.float32)
    pupil_valid = (np.isfinite(pupil) & (pupil > 0)) if pupil_valid is None else pupil_valid

    left_open = np.zeros(n, np.float32) if left_open is None else left_open.astype(np.float32)
    right_open = np.zeros(n, np.float32) if right_open is None else right_open.astype(np.float32)
    aoi_id = np.full(n, -1, np.int32) if aoi_id is None else aoi_id.astype(np.int32)

    open_mean = 0.5 * (left_open + right_open)

    x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4] = sx, sy, dx, dy, speed
    x[:, 5:8], x[:, 8] = gaze_dir[:, :3], dir_speed
    x[:, 9], x[:, 10], x[:, 11], x[:, 12] = left_open, right_open, open_mean, np.abs(left_open - right_open)
    x[:, 13], x[:, 14], x[:, 15], x[:, 16] = np.where(pupil_valid, pupil, 0), pupil_valid.astype(np.float32), tracking, hit

    for i, aid in enumerate(aoi_id):
        if 0 <= int(aid) < AOI_COUNT:
            x[i, 17 + int(aid)] = 1.0

    append_window_features(x, pupil, pupil_valid, tracking, aoi_id, left_open, right_open, speed)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def parse_gazebasevr_task(path: Path) -> str:
    # Examples: S_1002_S1_1_VRG.csv -> VRG
    return path.stem.split("_")[-1].upper()


def parse_subject(stem: str) -> str:
    # Examples: S_1002_S1_1_VRG -> S_1002
    m = re.match(r"(S_\d+|[A-Za-z]+\d+)", stem)
    return m.group(1) if m else stem.split("_")[0]


def sample_window_starts(n: int, args, rng: random.Random) -> List[int]:
    if n < args.window_frames:
        return []
    starts = list(range(0, n - args.window_frames + 1, args.step_frames))
    if args.max_windows_per_file and len(starts) > args.max_windows_per_file:
        starts = sorted(rng.sample(starts, args.max_windows_per_file))
    return starts


def slice_windows(
    features: np.ndarray,
    y: np.ndarray,
    group: str,
    source: str,
    starts: List[int],
    args,
) -> List[Sample]:
    out = []
    dtype = np.float16 if args.storage_dtype == "float16" else np.float32

    for start in starts:
        end = start + args.window_frames
        if int(features[start:end, 15].sum()) < args.min_valid_frames:
            continue
        out.append(Sample(features[start:end].astype(dtype, copy=True), y.copy(), group, source))
    return out


def read_gazebasevr_file(path: Path, task_to_idx: Dict[str, int], args, rng: random.Random) -> List[Sample]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] skip unreadable file: {path.name}: {exc}")
        return []

    required = {"n", "x", "y", "lx", "ly", "rx", "ry"}
    if not required.issubset(df.columns):
        print(f"[WARN] skip {path.name}: missing columns {sorted(required - set(df.columns))}")
        return []

    step = max(1, int(round(args.source_hz / args.target_hz)))
    df = df.iloc[::step].reset_index(drop=True)
    if len(df) < args.window_frames:
        return []

    sx, sy = to_float(df["x"]), to_float(df["y"])
    ts = to_float(df["n"], 0.0)

    lx = to_float(df["lx"])
    ly = to_float(df["ly"])
    rx = to_float(df["rx"])
    ry = to_float(df["ry"])

    tracking = (
        np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sy >= 0) &
        np.isfinite(lx) & np.isfinite(ly) & np.isfinite(rx) & np.isfinite(ry)
    ).astype(np.float32)

    gaze_dir = np.zeros((len(df), 3), dtype=np.float32)
    if {"xT", "yT", "zT"}.issubset(df.columns):
        gaze_dir = np.stack([to_float(df[c], 0.0) for c in ("xT", "yT", "zT")], axis=1)

    # GazeBaseVR raw file here has no pupil diameter. Keep pupil invalid and
    # use tracking as a weak openness proxy for compatibility with finetune features.
    features = build_features(
        sx,
        sy,
        ts,
        tracking,
        tracking.copy(),
        pupil=np.zeros(len(df), np.float32),
        pupil_valid=np.zeros(len(df), dtype=bool),
        gaze_dir=gaze_dir,
        left_open=tracking,
        right_open=tracking,
        aoi_id=np.full(len(df), -1, dtype=np.int32),
    )

    task = parse_gazebasevr_task(path)
    if task not in task_to_idx:
        return []
    y = np.zeros(len(task_to_idx), dtype=np.float32)
    y[task_to_idx[task]] = 1.0

    starts = sample_window_starts(len(features), args, rng)
    return slice_windows(features, y, parse_subject(path.stem), str(path), starts, args)


def reservoir_extend(
    reservoir: List[Sample],
    incoming: List[Sample],
    max_total: int,
    rng: random.Random,
    seen_count: int,
) -> int:
    """Reservoir sampling at window level.

    Keeps at most max_total windows without first materializing all windows.
    """
    if max_total <= 0:
        reservoir.extend(incoming)
        return seen_count + len(incoming)

    for item in incoming:
        seen_count += 1
        if len(reservoir) < max_total:
            reservoir.append(item)
        else:
            j = rng.randint(0, seen_count - 1)
            if j < max_total:
                reservoir[j] = item
    return seen_count


def load_gazebasevr_samples(args):
    root = Path(args.gazebasevr_dir)
    if not root.exists():
        raise FileNotFoundError(f"GazeBaseVR directory not found: {root}")

    files = sorted(root.rglob("*.csv"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No CSV files found under: {root}")

    tasks = sorted({parse_gazebasevr_task(p) for p in files})
    task_to_idx = {task: i for i, task in enumerate(tasks)}
    print(f"[INFO] found {len(files)} csv files")
    print(f"[INFO] detected tasks: {task_to_idx}")

    rng = random.Random(args.seed)
    samples: List[Sample] = []
    seen_windows = 0
    usable_files = 0
    skipped_files = 0

    for i, path in enumerate(files, start=1):
        new_samples = read_gazebasevr_file(path, task_to_idx, args, rng)
        if new_samples:
            usable_files += 1
            seen_windows = reservoir_extend(samples, new_samples, args.max_total_windows, rng, seen_windows)
        else:
            skipped_files += 1

        if i % 50 == 0 or i == len(files):
            print(
                f"[INFO] processed {i}/{len(files)} files | "
                f"kept_windows={len(samples)} | seen_windows={seen_windows} | usable_files={usable_files}"
            )

    if not samples:
        raise RuntimeError("No pretrain windows found. Check CSV format, window length, and min_valid_frames.")

    return samples, task_to_idx, {
        "num_files": len(files),
        "usable_files": usable_files,
        "skipped_files": skipped_files,
        "seen_windows_before_reservoir": seen_windows,
        "kept_windows": len(samples),
    }


def split_by_group(samples: List[Sample], seed: int):
    groups = sorted({s.group for s in samples})
    random.Random(seed).shuffle(groups)
    n = len(groups)

    if n <= 1:
        # Last fallback: random sample-level split.
        random.Random(seed).shuffle(samples)
        tr_end = max(1, int(len(samples) * 0.8))
        va_end = max(tr_end + 1, int(len(samples) * 0.9))
        return samples[:tr_end], samples[tr_end:va_end], samples[va_end:] or samples[tr_end:va_end]

    tr_end = max(1, int(round(n * 0.70)))
    va_end = min(n, tr_end + max(1, int(round(n * 0.15))))
    tr, va, te = set(groups[:tr_end]), set(groups[tr_end:va_end]), set(groups[va_end:])

    if not va and len(tr) > 1:
        va.add(tr.pop())
    if not te and va:
        te.add(va.pop())

    train = [s for s in samples if s.group in tr]
    val = [s for s in samples if s.group in va] or train[: max(1, len(train) // 10)]
    test = [s for s in samples if s.group in te] or val
    return train, val, test


def compute_stats_streaming(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    """Streaming mean/std to avoid building a giant [N, FEATURE_DIM] array."""
    mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    std = np.ones(FEATURE_DIM, dtype=np.float32)

    sum_vec = np.zeros(len(CONTINUOUS_IDXS), dtype=np.float64)
    sumsq_vec = np.zeros(len(CONTINUOUS_IDXS), dtype=np.float64)
    count = 0

    for s in samples:
        x = s.x[:, CONTINUOUS_IDXS].astype(np.float64, copy=False)
        sum_vec += x.sum(axis=0)
        sumsq_vec += (x * x).sum(axis=0)
        count += x.shape[0]

    if count <= 0:
        return mean, std

    cont_mean = sum_vec / count
    cont_var = sumsq_vec / count - cont_mean ** 2
    cont_var = np.maximum(cont_var, 1e-6)
    cont_std = np.sqrt(cont_var)

    mean[CONTINUOUS_IDXS] = cont_mean.astype(np.float32)
    std[CONTINUOUS_IDXS] = cont_std.astype(np.float32)
    std[np.abs(std) < 1e-6] = 1.0
    return mean, std


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total = 0
    correct = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        truth = yb.argmax(dim=1)
        correct += int((pred == truth).sum().item())
        total += int(xb.size(0))

    return {"accuracy": float(correct / max(1, total))}


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, task_to_idx, data_info = load_gazebasevr_samples(args)
    train_s, val_s, test_s = split_by_group(samples, args.seed)
    print(f"[INFO] split windows: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    mean, std = compute_stats_streaming(train_s)

    train_loader = DataLoader(
        EyeDataset(train_s, mean, std),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        EyeDataset(val_s, mean, std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )
    test_loader = DataLoader(
        EyeDataset(test_s, mean, std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )

    model = TinyEyeTransformer(
        num_classes=len(task_to_idx),
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    best_path = out_dir / "eye_transformer_pretrained.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb.argmax(dim=1).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * int(xb.size(0))

        val_metrics = evaluate(model, val_loader, args.device)
        train_loss = total_loss / max(1, len(train_s))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "stage": "pretrain",
                    "model_state": model.state_dict(),
                    "feature_names": FEATURE_NAMES,
                    "label_map": task_to_idx,
                    "args": vars(args),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, args.device)

    # Save a second copy with a more explicit name. The finetune script can load either.
    backbone_path = out_dir / "gazebasevr_pretrained_backbone.pt"
    torch.save(ckpt, backbone_path)

    write_json(out_dir / "pretrain_history.json", history)
    write_json(
        out_dir / "pretrain_stats.json",
        {
            "dataset": "GazeBaseVR",
            "task_to_idx": task_to_idx,
            "data_info": data_info,
            "split": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
            "test_metrics": test_metrics,
            "feature_names": FEATURE_NAMES,
            "continuous_feature_idxs": CONTINUOUS_IDXS,
            "binary_feature_idxs": BINARY_IDXS,
            "onehot_feature_idxs": ONEHOT_IDXS,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "saved_checkpoint": str(best_path),
            "saved_backbone": str(backbone_path),
        },
    )

    print(f"[DONE] saved checkpoint: {best_path}")
    print(f"[DONE] saved backbone copy: {backbone_path}")
    print(f"[DONE] test metrics: {json.dumps(test_metrics, ensure_ascii=False)}")


def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
