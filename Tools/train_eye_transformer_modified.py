#!/usr/bin/env python3
"""Offline TinyEyeTransformer training for GazeBaseVR pretrain + Unity finetune.

Unity loads only the exported ONNX and stats JSON. No runtime training happens in
the Unity scene.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score
except Exception:
    RandomForestClassifier = None
    f1_score = None
    accuracy_score = None


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
CONTINUOUS_IDXS = list(range(0, 14)) + list(range(27, 47))
BINARY_IDXS = [14, 15, 16]
ONEHOT_IDXS = list(range(17, 27))
WINDOW_IDXS = list(range(27, 47))
RF_FEATURE_NAMES = [
    "valid_ratio", "aoi_transition_rate", "aoi_entropy", "speed_p50", "speed_p90",
    "screen_x_mean", "screen_y_mean", "screen_x_std", "screen_y_std",
    "openness_mean", "openness_std", "blink_ratio",
] + [f"aoi_dwell_{i}" for i in range(AOI_COUNT)] + [f"aoi_first_order_{i}" for i in range(AOI_COUNT)]


@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray
    group: str
    source: str


class EyeDataset(Dataset):
    def __init__(self, samples, mean, std):
        self.samples = list(samples)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = s.x.astype(np.float32, copy=True)
        denom = np.where(np.abs(self.std) < 1e-6, 1.0, self.std)
        x[:, CONTINUOUS_IDXS] = (x[:, CONTINUOUS_IDXS] - self.mean[CONTINUOUS_IDXS]) / denom[CONTINUOUS_IDXS]
        return torch.from_numpy(x), torch.from_numpy(s.y.astype(np.float32))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TinyEyeTransformer(nn.Module):
    def __init__(self, num_classes=AOI_COUNT, mode="finetune", d_model=96, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.proj = nn.Sequential(nn.Linear(FEATURE_DIM, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        if mode == "finetune":
            self.widget_query = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        else:
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def padding_mask(self, x):
        invalid = x[:, :, 15] <= 0.5
        all_invalid = invalid.all(dim=1)
        if all_invalid.any():
            invalid = invalid.clone()
            invalid[all_invalid, 0] = False
        return invalid

    def encode(self, x):
        mask = self.padding_mask(x)
        z = self.encoder(self.pos(self.proj(x)), src_key_padding_mask=mask)
        return z, mask

    def weighted_gap(self, z, mask):
        valid = (~mask).float().unsqueeze(-1)
        w = self.gate(z) * valid
        return (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)

    def forward(self, x):
        z, mask = self.encode(x)
        pooled = self.weighted_gap(z, mask)
        if self.mode != "finetune":
            return self.head(pooled)
        q = self.widget_query.unsqueeze(0).expand(x.size(0), -1, -1)
        attn, _ = self.cross_attn(q, z, z, key_padding_mask=mask, need_weights=False)
        pooled = pooled.unsqueeze(1).expand(-1, self.num_classes, -1)
        return self.head(torch.cat([attn, pooled], dim=-1)).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, y):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(y > 0.5, p, 1.0 - p).clamp(1e-6, 1.0)
        alpha = torch.where(y > 0.5, torch.full_like(y, self.alpha), torch.full_like(y, 1.0 - self.alpha))
        return (alpha * (1.0 - pt).pow(self.gamma) * bce).mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--gazebasevr_dir", default="")
    p.add_argument("--frame_dir", default=r"F:\TestData")
    p.add_argument("--out_dir", default=r"F:\TestData\OUT_DIR")
    p.add_argument("--pretrained_ckpt", default="")
    p.add_argument("--export_onnx", default="")
    p.add_argument("--stats_json", default="")
    p.add_argument("--target_hz", type=int, default=50)
    p.add_argument("--source_hz", type=int, default=250)
    p.add_argument("--window_frames", type=int, default=150)
    p.add_argument("--step_frames", type=int, default=15)
    p.add_argument("--min_valid_frames", type=int, default=20)
    p.add_argument("--aoi_label_base", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_windows_per_file", type=int, default=0)
    p.add_argument("--loss", choices=["focal", "bce"], default="focal")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train_random_forest", action="store_true")
    p.add_argument("--export_rf_json", default="")
    p.add_argument("--rf_estimators", type=int, default=120)
    p.add_argument("--rf_max_depth", type=int, default=8)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float(series, default=np.nan):
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(np.float32)


def normalize_axis(values, valid):
    out = np.zeros_like(values, dtype=np.float32)
    ok = valid & np.isfinite(values)
    if not ok.any():
        return out
    vv = values[ok]
    if float(np.nanmin(vv)) >= 0 and float(np.nanmax(vv)) <= 1:
        out[ok] = np.clip(vv, 0, 1)
        return out
    lo, hi = np.percentile(vv, [1, 99])
    if abs(float(hi - lo)) < 1e-6:
        lo, hi = float(np.nanmin(vv)), float(np.nanmax(vv))
    if abs(float(hi - lo)) >= 1e-6:
        out[ok] = np.clip((vv - lo) / (hi - lo), 0, 1)
    return out


def build_features(sx_raw, sy_raw, ts_ms, tracking, hit, pupil=None, pupil_valid=None, gaze_dir=None, left_open=None, right_open=None, aoi_id=None):
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


def append_window_features(x, pupil, pupil_valid, tracking, aoi_id, left_open, right_open, speed):
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


def parse_targets(value, base):
    y = np.zeros(AOI_COUNT, dtype=np.float32)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return y
    for token in re.split(r"[|,;\s]+", str(value)):
        if not token:
            continue
        try:
            idx = int(float(token)) - base
        except ValueError:
            continue
        if 0 <= idx < AOI_COUNT:
            y[idx] = 1.0
    return y


def slice_windows(features, y, group, source, args):
    samples = []
    starts = list(range(0, len(features) - args.window_frames + 1, args.step_frames))
    if args.max_windows_per_file and len(starts) > args.max_windows_per_file:
        starts = starts[: args.max_windows_per_file]
    for start in starts:
        end = start + args.window_frames
        if int(features[start:end, 15].sum()) < args.min_valid_frames:
            continue
        samples.append(Sample(features[start:end], y.copy(), group, source))
    return samples


def parse_gazebasevr_task(path):
    return path.stem.split("_")[-1].upper()


def parse_subject(stem):
    m = re.match(r"(S_\d+|[A-Za-z]+\d+)", stem)
    return m.group(1) if m else stem.split("_")[0]


def read_gazebasevr(path, task_to_idx, args):
    df = pd.read_csv(path)
    if not {"n", "x", "y", "lx", "ly", "rx", "ry"}.issubset(df.columns):
        return []
    step = max(1, int(round(args.source_hz / args.target_hz)))
    df = df.iloc[::step].reset_index(drop=True)
    if len(df) < args.window_frames:
        return []
    sx, sy = to_float(df["x"]), to_float(df["y"])
    ts = to_float(df["n"], 0.0)
    tracking = (
        np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sy >= 0) &
        np.isfinite(to_float(df["lx"])) & np.isfinite(to_float(df["rx"]))
    ).astype(np.float32)
    gaze_dir = np.zeros((len(df), 3), dtype=np.float32)
    if {"xT", "yT", "zT"}.issubset(df.columns):
        gaze_dir = np.stack([to_float(df[c], 0.0) for c in ("xT", "yT", "zT")], axis=1)
    features = build_features(
        sx, sy, ts, tracking, tracking.copy(), gaze_dir=gaze_dir,
        pupil=np.zeros(len(df), np.float32), pupil_valid=np.zeros(len(df), bool),
        left_open=tracking, right_open=tracking,
    )
    y = np.zeros(len(task_to_idx), dtype=np.float32)
    y[task_to_idx[parse_gazebasevr_task(path)]] = 1.0
    return slice_windows(features, y, parse_subject(path.stem), str(path), args)


def select_unity_label(window, base):
    non_empty = window["target_controls"].dropna().astype(str)
    non_empty = non_empty[non_empty.str.len() > 0]
    return parse_targets(non_empty.iloc[-1], base) if len(non_empty) else np.zeros(AOI_COUNT, dtype=np.float32)


def read_unity_frame(path, args):
    df = pd.read_csv(path)
    if "screen_x_norm" not in df.columns or "target_controls" not in df.columns:
        return []
    n = len(df)
    sx, sy = to_float(df["screen_x_norm"]), to_float(df["screen_y_norm"])
    ts = to_float(df.get("timestamp_ms", pd.Series(np.arange(n))), 0.0)
    tracking = (to_float(df.get("tracking_state", pd.Series(np.ones(n))), 0.0) > 0).astype(np.float32)
    hit = (to_float(df.get("hit_ui_plane", pd.Series(np.ones(n))), 0.0) > 0).astype(np.float32)
    gaze_dir = np.stack([
        to_float(df.get("gaze_dir_x", pd.Series(np.zeros(n))), 0.0),
        to_float(df.get("gaze_dir_y", pd.Series(np.zeros(n))), 0.0),
        to_float(df.get("gaze_dir_z", pd.Series(np.zeros(n))), 0.0),
    ], axis=1)
    pupil = to_float(df.get("pupil_diameter", pd.Series(np.zeros(n))), 0.0)
    pupil_valid = to_float(df.get("pupil_valid", pd.Series((pupil > 0).astype(int))), 0.0) > 0.5
    left_open = to_float(df.get("left_openness", pd.Series(np.zeros(n))), 0.0)
    right_open = to_float(df.get("right_openness", pd.Series(np.zeros(n))), 0.0)
    aoi_id = pd.to_numeric(df.get("aoi_id", pd.Series(np.full(n, -1))), errors="coerce").fillna(-1).to_numpy(np.int32)
    features = build_features(sx, sy, ts, tracking, hit, pupil, pupil_valid, gaze_dir, left_open, right_open, aoi_id)
    group = str(df["subject_id"].dropna().iloc[0]) if "subject_id" in df and df["subject_id"].notna().any() else path.stem
    samples = []
    starts = list(range(0, len(df) - args.window_frames + 1, args.step_frames))
    if args.max_windows_per_file and len(starts) > args.max_windows_per_file:
        starts = starts[: args.max_windows_per_file]
    for start in starts:
        end = start + args.window_frames
        if int(features[start:end, 15].sum()) < args.min_valid_frames:
            continue
        y = select_unity_label(df.iloc[start:end], args.aoi_label_base)
        if y.sum() > 0:
            samples.append(Sample(features[start:end], y, group, str(path)))
    return samples


def load_samples(args):
    samples = []
    if args.stage == "pretrain":
        files = sorted(Path(args.gazebasevr_dir).rglob("*.csv"))
        if args.max_files:
            files = files[: args.max_files]
        task_to_idx = {task: i for i, task in enumerate(sorted({parse_gazebasevr_task(p) for p in files}))}
        for path in files:
            samples.extend(read_gazebasevr(path, task_to_idx, args))
        labels = task_to_idx
    else:
        files = sorted(Path(args.frame_dir).rglob("Frame_*.csv"))
        if args.max_files:
            files = files[: args.max_files]
        for path in files:
            samples.extend(read_unity_frame(path, args))
        labels = {f"Control_{i}": i for i in range(AOI_COUNT)}
    if not samples:
        raise RuntimeError(f"No windows found for stage={args.stage}")
    return samples, labels


def split_by_group(samples, seed):
    groups = sorted({s.group for s in samples})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    tr_end = max(1, int(round(n * 0.70))) if n > 2 else max(1, n - 1)
    va_end = min(n, tr_end + max(1, int(round(n * 0.15)))) if n > 2 else tr_end
    tr, va, te = set(groups[:tr_end]), set(groups[tr_end:va_end]), set(groups[va_end:])
    if not te and va:
        te.add(va.pop())
    train = [s for s in samples if s.group in tr]
    val = [s for s in samples if s.group in va] or train[: max(1, len(train) // 10)]
    test = [s for s in samples if s.group in te] or val
    return train, val, test


def compute_stats(samples):
    values = np.concatenate([s.x.reshape(-1, FEATURE_DIM) for s in samples], axis=0)
    mean, std = np.zeros(FEATURE_DIM, np.float32), np.ones(FEATURE_DIM, np.float32)
    mean[CONTINUOUS_IDXS] = values[:, CONTINUOUS_IDXS].mean(axis=0)
    std[CONTINUOUS_IDXS] = values[:, CONTINUOUS_IDXS].std(axis=0)
    std[np.abs(std) < 1e-6] = 1.0
    return mean, std


def rf_window_features(sample):
    x = sample.x
    tracking = x[:, 15] > 0.5
    hit = x[:, 16] > 0.5
    coord_valid = tracking & hit & np.isfinite(x[:, 0]) & np.isfinite(x[:, 1])
    valid_count = int(coord_valid.sum())
    valid_ratio = valid_count / max(1, len(x))

    sx = x[coord_valid, 0]
    sy = x[coord_valid, 1]
    speed = x[coord_valid, 4]
    openness = x[coord_valid, 11]
    aoi_ids = np.argmax(x[:, 17:27], axis=1)
    has_aoi = (x[:, 17:27].sum(axis=1) > 0.5) & coord_valid
    valid_aoi = aoi_ids[has_aoi]

    dwell = np.zeros(AOI_COUNT, dtype=np.float32)
    first_order = np.zeros(AOI_COUNT, dtype=np.float32)
    if len(valid_aoi):
        for order, aid in enumerate(valid_aoi, start=1):
            dwell[int(aid)] += 1.0
            if first_order[int(aid)] <= 0:
                first_order[int(aid)] = order / max(1, len(valid_aoi))
        dwell /= float(len(valid_aoi))

    transitions = float(np.sum(valid_aoi[1:] != valid_aoi[:-1])) / max(1, len(valid_aoi) - 1) if len(valid_aoi) > 1 else 0.0
    nonzero = dwell[dwell > 0]
    entropy = float(-(nonzero * np.log(nonzero + 1e-8)).sum()) if len(nonzero) else 0.0

    open_valid = openness[np.isfinite(openness)]
    features = [
        valid_ratio,
        transitions,
        entropy,
        float(np.percentile(speed, 50)) if len(speed) else 0.0,
        float(np.percentile(speed, 90)) if len(speed) else 0.0,
        float(np.mean(sx)) if len(sx) else 0.0,
        float(np.mean(sy)) if len(sy) else 0.0,
        float(np.std(sx)) if len(sx) else 0.0,
        float(np.std(sy)) if len(sy) else 0.0,
        float(np.mean(open_valid)) if len(open_valid) else 0.0,
        float(np.std(open_valid)) if len(open_valid) else 0.0,
        float(np.mean(open_valid < 0.2)) if len(open_valid) else 0.0,
    ]
    features.extend(dwell.tolist())
    features.extend(first_order.tolist())
    return np.asarray(features, dtype=np.float32)


def make_rf_matrix(samples):
    return np.stack([rf_window_features(s) for s in samples], axis=0), np.stack([s.y for s in samples], axis=0).astype(np.int32)


def evaluate_rf(models, samples, threshold):
    if not samples:
        return {"subset_accuracy": 0.0, "macro_f1": 0.0, "hit3": 0.0}
    x, y = make_rf_matrix(samples)
    probs = predict_rf(models, x)
    pred = (probs >= threshold).astype(np.int32)
    subset = float(np.mean(np.all(pred == y, axis=1)))
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0)) if f1_score else 0.0
    top3 = np.argsort(-probs, axis=1)[:, :3]
    hit3 = []
    for i in range(y.shape[0]):
        truth = np.where(y[i] > 0)[0].tolist()
        hit3.append(any(t in top3[i].tolist() for t in truth))
    return {"subset_accuracy": subset, "macro_f1": macro_f1, "hit3": float(np.mean(hit3)) if hit3 else 0.0}


def predict_rf(models, x):
    probs = np.zeros((x.shape[0], AOI_COUNT), dtype=np.float32)
    for class_idx, model in enumerate(models):
        if model is None:
            continue
        if len(getattr(model, "classes_", [])) == 1:
            probs[:, class_idx] = 1.0 if int(model.classes_[0]) == 1 else 0.0
            continue
        p = model.predict_proba(x)
        positive_col = list(model.classes_).index(1) if 1 in model.classes_ else -1
        probs[:, class_idx] = p[:, positive_col] if positive_col >= 0 else 0.0
    return probs


def train_random_forest(args, train_s, val_s, test_s, out_dir):
    if args.stage != "finetune":
        print("Random forest export is only defined for Unity finetune data; skipping.")
        return
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required for --train_random_forest")

    x_train, y_train = make_rf_matrix(train_s)
    models = []
    for class_idx in range(AOI_COUNT):
        clf = RandomForestClassifier(
            n_estimators=max(1, args.rf_estimators),
            max_depth=None if args.rf_max_depth <= 0 else args.rf_max_depth,
            random_state=args.seed + class_idx,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        clf.fit(x_train, y_train[:, class_idx])
        models.append(clf)

    val_metrics = evaluate_rf(models, val_s, args.threshold)
    test_metrics = evaluate_rf(models, test_s, args.threshold)
    write_json(out_dir / "random_forest_val_metrics.json", val_metrics)
    write_json(out_dir / "random_forest_test_metrics.json", test_metrics)

    export_path = Path(args.export_rf_json) if args.export_rf_json else out_dir / "eye_random_forest.json"
    export_random_forest_json(export_path, args, models, test_metrics)
    print(f"Exported random forest JSON: {export_path}")
    print(f"Random forest test metrics: {json.dumps(test_metrics, ensure_ascii=False)}")


def export_random_forest_json(path, args, models, metrics):
    forests = []
    for clf in models:
        trees = []
        for estimator in clf.estimators_:
            tree = estimator.tree_
            values = []
            positive_index = list(estimator.classes_).index(1) if 1 in estimator.classes_ else -1
            for node_value in tree.value:
                counts = node_value[0]
                denom = float(np.sum(counts))
                if denom <= 0 or positive_index < 0:
                    values.append(0.0)
                else:
                    values.append(float(counts[positive_index] / denom))
            trees.append({
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "left": tree.children_left.astype(int).tolist(),
                "right": tree.children_right.astype(int).tolist(),
                "value": values,
            })
        forests.append({"trees": trees})

    payload = {
        "model_type": "unity_random_forest_multilabel",
        "feature_names": RF_FEATURE_NAMES,
        "window_frames": args.window_frames,
        "step_frames": args.step_frames,
        "min_valid_frames": args.min_valid_frames,
        "early_min_valid_frames": max(args.min_valid_frames, 40),
        "aoi_count": AOI_COUNT,
        "threshold": args.threshold,
        "metrics": metrics,
        "forests": forests,
    }
    write_json(path, payload)


def make_model(args, num_classes):
    return TinyEyeTransformer(
        num_classes=num_classes, mode=args.stage, d_model=args.d_model,
        nhead=args.nhead, layers=args.layers, dropout=args.dropout
    )


def load_pretrained(model, ckpt_path):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} compatible tensors from {ckpt_path}")


@torch.no_grad()
def evaluate(model, loader, args):
    model.eval()
    logits, targets = [], []
    for xb, yb in loader:
        logits.append(model(xb.to(args.device)).cpu())
        targets.append(yb.cpu())
    logits, targets = torch.cat(logits, 0), torch.cat(targets, 0)
    if args.stage == "pretrain":
        pred = logits.argmax(1)
        truth = targets.argmax(1)
        return {"accuracy": float((pred == truth).float().mean().item())}
    probs = torch.sigmoid(logits)
    pred = (probs >= args.threshold).float()
    subset = (pred == targets).all(1).float().mean().item()
    tp = (pred * targets).sum(0)
    fp = (pred * (1 - targets)).sum(0)
    fn = ((1 - pred) * targets).sum(0)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    top3 = probs.topk(k=min(3, probs.size(1)), dim=1).indices
    hit3 = []
    for i in range(targets.size(0)):
        truth = torch.where(targets[i] > 0.5)[0].tolist()
        hit3.append(any(t in top3[i].tolist() for t in truth))
    return {
        "subset_accuracy": float(subset),
        "macro_f1": float(f1.mean().item()),
        "recall": float(recall.mean().item()),
        "hit3": float(np.mean(hit3)) if hit3 else 0.0,
    }


def write_metrics(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_unity_stats(path, args, mean, std, samples):
    labels = np.stack([s.y for s in samples], axis=0)
    stats = {
        "stage": "finetune",
        "dataset": "unity_frame_csv",
        "feature_names": FEATURE_NAMES,
        "active_feature_names": FEATURE_NAMES,
        "dropped_feature_names": [],
        "continuous_feature_idxs": CONTINUOUS_IDXS,
        "binary_feature_idxs": BINARY_IDXS,
        "onehot_feature_idxs": ONEHOT_IDXS,
        "window_feature_idxs": WINDOW_IDXS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "label_priors": labels.mean(axis=0).tolist(),
        "prior_strength": 0.0,
        "threshold": args.threshold,
        "window_frames": args.window_frames,
        "step_frames": args.step_frames,
        "min_valid_frames": args.min_valid_frames,
        "feature_dim": FEATURE_DIM,
        "aoi_count": AOI_COUNT,
        "aoi_label_base": args.aoi_label_base,
        "aoi_ids": list(range(AOI_COUNT)),
        "pretrained_ckpt": args.pretrained_ckpt,
        "use_aoi_features": True,
        "drop_empty_pupil_features": False,
        "top_k": 3,
        "onnx_opset": 18,
        "onnx_export_error": "",
    }
    write_json(path, stats)


def export_onnx(model, args, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, args.window_frames, FEATURE_DIM, dtype=torch.float32, device=args.device)
    dummy[:, :, 15] = 1.0
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(path), input_names=["gaze_window"], output_names=["logits"],
            dynamic_axes=None, opset_version=18
        )
    print(f"Exported ONNX: {path}")


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, label_map = load_samples(args)
    train_s, val_s, test_s = split_by_group(samples, args.seed)
    mean, std = compute_stats(train_s)
    if args.train_random_forest:
        train_random_forest(args, train_s, val_s, test_s, out_dir)
    num_classes = len(label_map) if args.stage == "pretrain" else AOI_COUNT
    model = make_model(args, num_classes).to(args.device)
    if args.stage == "finetune":
        load_pretrained(model, args.pretrained_ckpt)
    train_loader = DataLoader(EyeDataset(train_s, mean, std), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EyeDataset(val_s, mean, std), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EyeDataset(test_s, mean, std), batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss() if args.stage == "pretrain" else (FocalLoss() if args.loss == "focal" else nn.BCEWithLogitsLoss())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_metric = -1.0
    ckpt_path = out_dir / ("eye_transformer_pretrained.pt" if args.stage == "pretrain" else "eye_transformer_finetuned.pt")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb.argmax(1).long()) if args.stage == "pretrain" else criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
        val = evaluate(model, val_loader, args)
        metric = val["accuracy"] if args.stage == "pretrain" else val["macro_f1"]
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(train_s)), **val}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if metric > best_metric:
            best_metric = metric
            torch.save({
                "model_state": model.state_dict(),
                "feature_names": FEATURE_NAMES,
                "label_map": label_map,
                "args": vars(args),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, test_loader, args)
    write_metrics(out_dir / f"{args.stage}_history.csv", history)
    write_json(out_dir / f"{args.stage}_test_metrics.json", test)
    if args.stage == "finetune":
        write_unity_stats(Path(args.stats_json) if args.stats_json else out_dir / "eye_transformer_stats.json", args, mean, std, samples)
        if args.export_onnx:
            export_onnx(model, args, Path(args.export_onnx))
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Test metrics: {json.dumps(test, ensure_ascii=False)}")


def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
