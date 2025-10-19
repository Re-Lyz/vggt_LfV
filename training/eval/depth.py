# eval/depth.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import json, csv
import cv2  # 仅用于 resize（与评估一致的双线性/最近邻）

# ---------- I/O ----------
def _save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _save_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

# ---------- utils ----------
def _resize_to(src: np.ndarray, h: int, w: int, linear: bool = True) -> np.ndarray:
    """将 src resize 到 (h,w)。linear=True 用双线性，否则最近邻。"""
    if src.shape[:2] == (h, w):
        return src
    inter = cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST
    return cv2.resize(src, (w, h), interpolation=inter)

def _to_np_depth(x):
    """torch.Tensor 或 np.ndarray -> np.ndarray(float32)（不改变形状）"""
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _as_depth_list(x) -> Optional[List[np.ndarray]]:
    """
    统一将输入转为 List[np.ndarray(H,W)], dtype=float32。
    支持：None / List[H,W] / [S,H,W] / [B,T,H,W] / torch.Tensor 同形状。
    """
    if x is None:
        return None
    x_np = _to_np_depth(x)
    if isinstance(x, (list, tuple)):  # List[H,W]
        out = []
        for item in x_np:
            arr = _to_np_depth(item)
            if arr.ndim != 2:
                raise ValueError(f"Depth item must be HxW, got {arr.shape}")
            out.append(arr.astype(np.float32, copy=False))
        return out
    if x_np.ndim == 2:   # [H,W]
        return [x_np.astype(np.float32, copy=False)]
    if x_np.ndim == 3:   # [S,H,W]
        return [x_np[i].astype(np.float32, copy=False) for i in range(x_np.shape[0])]
    if x_np.ndim == 4:   # [B,T,H,W] -> 展平为 [S,H,W]
        BT = x_np.shape[0] * x_np.shape[1]
        x_np = x_np.reshape(BT, *x_np.shape[-2:])
        return [x_np[i].astype(np.float32, copy=False) for i in range(x_np.shape[0])]
    raise ValueError(f"Unsupported depth shape: {x_np.shape}")

# ---------- metrics ----------
def _depth_metrics_single(
    pred: np.ndarray,
    gt:   np.ndarray,
    mask: Optional[np.ndarray],
    align: str = "median",          # "median" | "mean" | "lsq" | "none"
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    clamp_after_align: bool = True, # 对齐后将深度裁到[min_depth,max_depth]
) -> Dict[str, float]:
    """
    计算常见深度评估指标（AbsRel / RMSE / δ1, δ2, δ3）。
    - 会将 pred resize 到 gt 的尺寸（双线性）；
    - 有效像素条件：gt 有效 & pred 有效 & (可选)外部 mask；
    - 对齐支持 median/mean/lsq/none；
    - 返回 dict，若无有效像素则各项为 NaN。
    """
    assert gt.ndim == 2, "gt must be HxW"
    H, W = gt.shape

    # 1) resize + 转 float64 以减少数值误差
    pred = _resize_to(pred, H, W, linear=True).astype(np.float64, copy=False)
    gt   = gt.astype(np.float64, copy=False)

    eps = 1e-8

    # 2) 有效像素筛选：同时约束 gt 和 pred
    valid = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    valid &= np.isfinite(pred) & (pred > min_depth)
    if mask is not None:
        # 约定：mask>0 表示有效像素（若你的 mask 语义相反请在上游转换）
        valid &= (mask > 0)

    if valid.sum() == 0:
        return {"abs_rel": np.nan, "rmse": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}

    pv = pred[valid]
    gv = gt[valid]

    # 3) 对齐（尺度）
    if align == "median":
        s = np.median(gv) / (np.median(pv) + eps)
        pv = pv * s
    elif align == "mean":
        s = np.mean(gv) / (np.mean(pv) + eps)
        pv = pv * s
    elif align == "lsq":
        # 最小二乘比例：argmin_s || s*pv - gv ||^2 -> s = (pv·gv)/(pv·pv)
        denom = float(np.dot(pv, pv)) + eps
        s = float(np.dot(pv, gv)) / denom
        pv = pv * s
    elif align == "none":
        pass
    else:
        raise ValueError(f"Unknown align mode: {align}")

    # 4) 可选：对齐后裁剪范围，避免极端值影响 RMSE/AbsRel
    if clamp_after_align:
        pv = np.clip(pv, min_depth, max_depth)
        gv = np.clip(gv, min_depth, max_depth)

    # 5) 指标
    abs_rel = float(np.mean(np.abs(pv - gv) / (gv + eps)))
    rmse    = float(np.sqrt(np.mean((pv - gv) ** 2)))

    ratio = np.maximum(pv / (gv + eps), gv / (pv + eps))
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < (1.25 ** 2)))
    d3 = float(np.mean(ratio < (1.25 ** 3)))

    return {"abs_rel": abs_rel, "rmse": rmse, "d1": d1, "d2": d2, "d3": d3}

# ---------- public APIs ----------
def evaluate_sequence_depth(
    pred_depth,                 # List[H,W] | [S,H,W] | [B,T,H,W] | torch.Tensor 同形状
    gt_depth,                   # 同上
    depth_mask=None,            # 可选：同上；语义：>0 为有效像素
    depth_cfg: Dict[str, Any] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    逐序列深度评估（AbsRel / RMSE / δ1/δ2/δ3）。
    - 直接使用内存中的 pred_depth / gt_depth（不读文件，不依赖 outputs）
    - 自动展平 [B,T,H,W] → [S,H,W]，或接受 List[H,W]
    - mask 语义：>0 为有效像素（若你的语义相反，请在上游翻转）
    - align: 'median' | 'mean' | 'lsq' | 'none'
    """
    if depth_cfg is None:
        depth_cfg = {}
    align = str(depth_cfg.get("align", "median"))
    min_d = float(depth_cfg.get("min_depth", 1e-3))
    max_d = float(depth_cfg.get("max_depth", 80.0))

    # 统一成 List[H,W]
    pred_list = _as_depth_list(pred_depth)
    gt_list   = _as_depth_list(gt_depth)
    m_list    = _as_depth_list(depth_mask) if depth_mask is not None else None

    if pred_list is None or gt_list is None:
        return {"abs_rel": np.nan, "rmse": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}

    S = min(len(pred_list), len(gt_list))
    if S == 0:
        return {"abs_rel": np.nan, "rmse": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}

    rows, per_frame = [], []
    for i in range(S):
        pd = pred_list[i]
        gd = gt_list[i]
        mk = m_list[i] if (m_list is not None and i < len(m_list)) else None
        # 将 mask 二值化为 {0,1}（>0 为有效）
        if mk is not None:
            mk = (mk > 0).astype(np.uint8)
        m = _depth_metrics_single(pd, gd, mk, align=align, min_depth=min_d, max_depth=max_d)
        per_frame.append(m)
        rows.append({"frame": i, **m})

    # per-frame CSV
    if out_dir is not None and rows:
        _save_csv(rows, Path(out_dir) / "depth_perframe.csv")

    # 聚合
    if not per_frame:
        seq_mean = {"abs_rel": np.nan, "rmse": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}
    else:
        keys = per_frame[0].keys()
        seq_mean = {k: float(np.nanmean([m[k] for m in per_frame])) for k in keys}

    # JSON
    if out_dir is not None:
        _save_json({"sequence_mean": seq_mean, "per_frame": per_frame}, Path(out_dir) / "metrics_depth.json")
    return seq_mean

def aggregate_depth_metrics(per_seq_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """跨序列聚合（取 nanmean）"""
    if not per_seq_metrics:
        return {}
    keys = per_seq_metrics[0].keys()
    return {k: float(np.nanmean([m.get(k, np.nan) for m in per_seq_metrics])) for k in keys}
