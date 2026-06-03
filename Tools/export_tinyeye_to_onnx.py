#!/usr/bin/env python3
"""Standalone ONNX exporter for TinyEyeTransformer finetune checkpoints.

This script loads a saved .pt checkpoint and exports the finetuned
TinyEyeTransformer model to ONNX. It uses an ONNX-friendly padding_mask()
implementation without Python data-dependent control flow.

Example:
python export_tinyeye_to_onnx.py ^
  --ckpt "F:\\TestData\\OUT_DIR_CALIBRATED\\eye_transformer_finetuned.pt" ^
  --onnx "F:\\TestData\\OUT_DIR_CALIBRATED\\eye_transformer_finetuned.onnx" ^
  --stats_json "F:\\TestData\\OUT_DIR_CALIBRATED\\eye_transformer_stats.json"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


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
    def __init__(
        self,
        num_classes: int = AOI_COUNT,
        mode: str = "finetune",
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
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

        if mode == "finetune":
            self.widget_query = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

    def padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return src_key_padding_mask with ONNX-friendly tensor logic.

        Original training code used:
            if all_invalid.any():
                invalid[all_invalid, 0] = False
        That Python branch depends on input data and breaks torch.export / ONNX.
        This implementation is mathematically equivalent but uses torch.where.
        """
        invalid = x[:, :, 15] <= 0.5  # [B, T], True means masked/invalid
        all_invalid = invalid.all(dim=1, keepdim=True)  # [B, 1]
        first_pos = (torch.arange(invalid.size(1), device=x.device).unsqueeze(0) == 0).expand_as(invalid)
        invalid = torch.where(all_invalid & first_pos, torch.zeros_like(invalid), invalid)
        return invalid

    def encode(self, x: torch.Tensor):
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
        if self.mode != "finetune":
            return self.head(pooled)
        q = self.widget_query.unsqueeze(0).expand(x.size(0), -1, -1)
        attn, _ = self.cross_attn(q, z, z, key_padding_mask=mask, need_weights=False)
        pooled = pooled.unsqueeze(1).expand(-1, self.num_classes, -1)
        return self.head(torch.cat([attn, pooled], dim=-1)).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser(description="Export TinyEyeTransformer .pt checkpoint to ONNX")
    p.add_argument("--ckpt", required=True, help="Path to eye_transformer_finetuned.pt")
    p.add_argument("--onnx", required=True, help="Output ONNX path")
    p.add_argument("--stats_json", default="", help="Optional stats JSON to update with ONNX info")
    p.add_argument("--window_frames", type=int, default=150)
    p.add_argument("--feature_dim", type=int, default=FEATURE_DIM)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dynamo", action="store_true", help="Use the new torch.export-based ONNX exporter. Default uses legacy exporter.")
    return p.parse_args()


def load_model_from_ckpt(ckpt_path: str, device: str) -> TinyEyeTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    d_model = int(saved_args.get("d_model", 96))
    nhead = int(saved_args.get("nhead", 4))
    layers = int(saved_args.get("layers", 2))
    dropout = float(saved_args.get("dropout", 0.1))
    num_classes = int(saved_args.get("num_classes", AOI_COUNT)) if "num_classes" in saved_args else AOI_COUNT

    model = TinyEyeTransformer(
        num_classes=num_classes,
        mode="finetune",
        d_model=d_model,
        nhead=nhead,
        layers=layers,
        dropout=dropout,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing tensors: {missing}")
    if unexpected:
        print(f"[WARN] unexpected tensors: {unexpected}")

    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Model config: d_model={d_model}, nhead={nhead}, layers={layers}, dropout={dropout}, num_classes={num_classes}")
    return model


def export_onnx(model: TinyEyeTransformer, args):
    out_path = Path(args.onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_dim != FEATURE_DIM:
        raise ValueError(f"feature_dim mismatch: got {args.feature_dim}, expected {FEATURE_DIM}")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dummy = torch.zeros(1, args.window_frames, FEATURE_DIM, dtype=torch.float32, device=device)
    # tracking_valid channel. Set valid frames to avoid all-masked dummy input.
    dummy[:, :, 15] = 1.0
    # hit_ui_plane channel.
    dummy[:, :, 16] = 1.0

    with torch.no_grad():
        # Quick forward test before export.
        y = model(dummy)
        print(f"Forward check OK. Output shape: {tuple(y.shape)}")

        export_kwargs = dict(
            model=model,
            args=(dummy,),
            f=str(out_path),
            input_names=["gaze_window"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=args.opset,
            do_constant_folding=True,
        )

        if args.dynamo:
            print("Exporting ONNX with new dynamo exporter...")
            torch.onnx.export(**export_kwargs, dynamo=True)
        else:
            print("Exporting ONNX with legacy exporter: dynamo=False ...")
            try:
                torch.onnx.export(**export_kwargs, dynamo=False)
            except TypeError:
                # Older PyTorch does not accept dynamo argument.
                torch.onnx.export(**export_kwargs)

    print(f"Exported ONNX: {out_path}")
    return out_path


def update_stats_json(stats_path: str, onnx_path: Path, args):
    if not stats_path:
        return
    path = Path(stats_path)
    if not path.exists():
        print(f"[WARN] stats_json not found, skip update: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["onnx_path"] = str(onnx_path)
    data["onnx_opset"] = args.opset
    data["onnx_export_error"] = ""
    data["window_frames"] = args.window_frames
    data["feature_dim"] = FEATURE_DIM
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated stats JSON: {path}")


def main():
    args = parse_args()
    model = load_model_from_ckpt(args.ckpt, args.device)
    onnx_path = export_onnx(model, args)
    update_stats_json(args.stats_json, onnx_path, args)


if __name__ == "__main__":
    main()
