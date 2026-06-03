#!/usr/bin/env python3
"""Export a trained TinyEyeTransformer checkpoint to ONNX.

This exporter avoids PyTorch's fused TransformerEncoderLayer ONNX issue by
rebuilding the same model with ONNX-friendly attention blocks and loading the
trained checkpoint weights.

Example:
python export_tinyeye_to_onnx_friendly.py ^
  --ckpt "F:\TestData\OUT_DIR_CALIBRATED\eye_transformer_finetuned.pt" ^
  --onnx "F:\TestData\OUT_DIR_CALIBRATED\eye_transformer_finetuned.onnx" ^
  --window_frames 150 ^
  --device cpu ^
  --verify
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ONNXFriendlySelfAttention(nn.Module):
    """Replacement for nn.MultiheadAttention self-attention.

    Parameter names intentionally match nn.MultiheadAttention where possible:
    in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias.
    """
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1.0e4)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class ONNXFriendlyCrossAttention(nn.Module):
    """Replacement for nn.MultiheadAttention cross-attention with compatible parameter names."""
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, nq, _ = query.shape
        seq_len = key_value.size(1)
        d = self.d_model
        q = F.linear(query, self.in_proj_weight[:d, :], self.in_proj_bias[:d])
        k = F.linear(key_value, self.in_proj_weight[d:2 * d, :], self.in_proj_bias[d:2 * d])
        v = F.linear(key_value, self.in_proj_weight[2 * d:, :], self.in_proj_bias[2 * d:])

        q = q.view(bsz, nq, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1.0e4)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, nq, self.d_model)
        return self.out_proj(out)


class ONNXFriendlyEncoderLayer(nn.Module):
    """Equivalent to TransformerEncoderLayer(norm_first=True, activation='gelu'), but without fused ops."""
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.self_attn = ONNXFriendlySelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src = src + self.self_attn(self.norm1(src), key_padding_mask=src_key_padding_mask)
        src = src + self.linear2(F.gelu(self.linear1(self.norm2(src))))
        return src


class ONNXFriendlyEncoder(nn.Module):
    """Container preserving checkpoint key pattern: encoder.layers.0..."""
    def __init__(self, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ONNXFriendlyEncoderLayer(d_model, nhead) for _ in range(layers)])

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return src


class TinyEyeTransformerONNX(nn.Module):
    """ONNX-friendly inference model matching the finetune architecture."""
    def __init__(self, num_classes: int, d_model: int, nhead: int, layers: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = PositionalEncoding(d_model)
        self.encoder = ONNXFriendlyEncoder(d_model, nhead, layers)
        self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.widget_query = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.cross_attn = ONNXFriendlyCrossAttention(d_model, nhead)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        invalid = x[:, :, 15] <= 0.5
        all_invalid = invalid.all(dim=1, keepdim=True)
        t = torch.arange(invalid.size(1), device=x.device).unsqueeze(0)
        first_pos = (t == 0).expand_as(invalid)
        return torch.where(all_invalid & first_pos, torch.zeros_like(invalid), invalid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.padding_mask(x)
        z = self.encoder(self.pos(self.proj(x)), src_key_padding_mask=mask)

        valid = (~mask).float().unsqueeze(-1)
        w = self.gate(z) * valid
        pooled = (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0e-6)

        q = self.widget_query.unsqueeze(0).expand(x.size(0), -1, -1)
        attn = self.cross_attn(q, z, key_padding_mask=mask)
        pooled = pooled.unsqueeze(1).expand(-1, self.widget_query.size(0), -1)

        return self.head(torch.cat([attn, pooled], dim=-1)).squeeze(-1)


def infer_config_from_checkpoint(state: dict, ckpt_payload: dict):
    d_model = int(state["proj.0.weight"].shape[0])
    num_classes = int(state["widget_query"].shape[0])

    layer_ids = set()
    for k in state:
        if k.startswith("encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    layers = max(layer_ids) + 1 if layer_ids else 2

    ckpt_args = ckpt_payload.get("args", {}) if isinstance(ckpt_payload, dict) else {}
    nhead = int(ckpt_args.get("nhead", 4))
    dropout = float(ckpt_args.get("dropout", 0.0))

    return d_model, num_classes, layers, nhead, dropout


def load_model(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    d_model, num_classes, layers, nhead, dropout = infer_config_from_checkpoint(state, ckpt if isinstance(ckpt, dict) else {})

    model = TinyEyeTransformerONNX(
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        layers=layers,
        dropout=dropout,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys while loading:")
        for k in missing:
            print("  -", k)
    if unexpected:
        print("[WARN] Unexpected keys while loading:")
        for k in unexpected:
            print("  -", k)

    model.to(args.device)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Model config: d_model={d_model}, nhead={nhead}, layers={layers}, dropout={dropout}, num_classes={num_classes}")
    return model


def export_onnx(model, args):
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, args.window_frames, FEATURE_DIM, dtype=torch.float32, device=args.device)
    dummy[:, :, 15] = 1.0

    with torch.no_grad():
        out = model(dummy)
    print(f"Forward check OK. Output shape: {tuple(out.shape)}")

    print("Exporting ONNX with ONNX-friendly custom Transformer blocks ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["gaze_window"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"Exported ONNX: {onnx_path}")
    return onnx_path


def verify_onnx(onnx_path, args):
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, args.window_frames, FEATURE_DIM), dtype=np.float32)
    x[:, :, 15] = 1.0
    y = sess.run(None, {"gaze_window": x})[0]
    print(f"ONNX Runtime check OK. Output shape: {y.shape}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to eye_transformer_finetuned.pt")
    p.add_argument("--onnx", required=True, help="Output ONNX path")
    p.add_argument("--window_frames", type=int, default=150)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device != "cpu":
        print("[WARN] ONNX export is usually more stable on CPU. Current device:", args.device)
    model = load_model(args)
    onnx_path = export_onnx(model, args)
    if args.verify:
        verify_onnx(onnx_path, args)


if __name__ == "__main__":
    main()
