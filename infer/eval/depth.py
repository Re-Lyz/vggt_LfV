# eval/depth.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import json, csv
import cv2
from PIL import Image

# ---------- I/O ----------
def _save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _save_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


# ---------- utils ----------
def _read_depth(
    path: str,
    scale_adjustment: float = 1.0,
    png_encoding: str = "half",     # "half": 16bit 按 float16 解释；"uint16": 当作物理量的 uint16
    nonpositive_to_zero: bool = False,
) -> np.ndarray:
    """
    统一读取深度图（.tif/.tiff/.exr/.png）为 float32，并与 read_depth / read_depth_any 的处理保持一致：
      - PNG: 默认 half-float 还原；也可选 uint16 物理量。
      - EXR: 读取首通道，并把 >1e9 的异常值置 0。
      - 所有格式：统一转 float32；非有限值置 0；可选把 ≤0 置 0；最后乘 scale_adjustment。
    """
    p = path.lower()
    if p.endswith((".tif", ".tiff")):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"Failed to read TIFF: {path}")
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32)

    elif p.endswith(".exr"):
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if d is None:
            raise ValueError(f"Failed to read EXR: {path}")
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32)
        # 与 read_depth 同步：极端大值粗暴清零
        d[d > 1e9] = 0.0

    elif p.endswith(".png"):
        if png_encoding == "half":
            # 与 load_16big_png_depth 完全一致的 half-float 解码
            with Image.open(path) as depth_pil:
                arr_u16 = np.array(depth_pil, dtype=np.uint16)    # H×W, uint16
            d = arr_u16.view(np.float16).astype(np.float32)       # H×W, float32
            # 如遇端序异常可启用： d = arr_u16.byteswap().view(np.float16).astype(np.float32)
        elif png_encoding == "uint16":
            with Image.open(path) as depth_pil:
                arr_u16 = np.array(depth_pil, dtype=np.uint16)
            d = arr_u16.astype(np.float32)
        else:
            raise ValueError(f"Unknown png_encoding: {png_encoding}")
    else:
        raise ValueError(f"Unsupported depth extension: {path}")

    # 统一清理与缩放（与 read_depth / read_depth_any 对齐）
    d = d.astype(np.float32, copy=False)
    d[~np.isfinite(d)] = 0.0
    if nonpositive_to_zero:
        d[d <= 0] = 0.0

    if scale_adjustment != 1.0:
        d *= scale_adjustment

    return d

def _read_mask(path: str, valid_when_zero: bool = True) -> np.ndarray:
    """
    读取遮挡/有效性掩膜并二值化为 {0,1} 的 uint8。
    默认语义：黑=0=有效；白>0=无效（valid_when_zero=True）。
    
    Returns:
        mask (np.uint8): 1 表示“有效像素”，0 表示“无效/遮挡”。
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)

    # 取单通道：优先 alpha；否则转灰度
    if m.ndim == 3 and m.shape[2] == 4:
        m = m[..., 3]  # alpha 通道
    elif m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    # 根据语义二值化
    if valid_when_zero:
        # 0 = 有效 → mask=1；>0 = 无效 → mask=0
        mask = (m == 0)
    else:
        # >0 = 有效 → mask=1；0 = 无效 → mask=0
        mask = (m > 0)

    return mask.astype(np.uint8)

def _resize_to(src: np.ndarray, h: int, w: int, linear: bool = True) -> np.ndarray:
    if src.shape[:2] == (h, w): return src
    inter = cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST
    return cv2.resize(src, (w, h), interpolation=inter)

def _to_np_depth(x):
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _depths_from_outputs(outputs: Dict[str, Any]) -> Optional[List[np.ndarray]]:
    """
    只接受 outputs['depth']，形状必须为 [B=1, S, H, W, 1]。
    返回：List[np.ndarray]，长度 S，每个元素形状 [H, W]（float32）
    """
    if "depth" not in outputs:
        raise KeyError("[depth] 需要 outputs['depth']（[1,S,H,W,1]），但未找到该键。")

    D = _to_np_depth(outputs["depth"])
    if D.ndim != 5 or D.shape[0] != 1 or D.shape[-1] != 1:
        raise ValueError(f"[depth] outputs['depth'] 形状必须为 [1,S,H,W,1]，当前为 {D.shape}。")

    D = D[0]               # [S, H, W, 1]
    D = D[..., 0]          # [S, H, W]
    S = D.shape[0]
    return [D[i].astype(np.float32, copy=False) for i in range(S)]


# ---------- metrics ----------
def _depth_metrics_single(
    pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray],
    align: str = "median", min_depth: float = 1e-3, max_depth: float = 80.0
) -> Dict[str, float]:
    H, W = gt.shape
    pred = _resize_to(pred, H, W, linear=True)

    valid = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    if mask is not None:
        valid &= (mask > 0)

    if valid.sum() == 0:
        return {"abs_rel": np.nan, "rmse": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}

    pv = pred[valid].astype(np.float64)
    gv = gt[valid].astype(np.float64)

    if align == "median":
        s = np.median(gv) / (np.median(pv) + 1e-12)
        pv *= s
    elif align == "mean":
        s = np.mean(gv) / (np.mean(pv) + 1e-12)
        pv *= s
    # align == "none": 不缩放

    abs_rel = float(np.mean(np.abs(pv - gv) / (gv + 1e-12)))
    rmse = float(np.sqrt(np.mean((pv - gv) ** 2)))
    ratio = np.maximum(pv / (gv + 1e-12), gv / (pv + 1e-12))
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))
    return {"abs_rel": abs_rel, "rmse": rmse, "d1": d1, "d2": d2, "d3": d3}


# ---------- public APIs ----------
def evaluate_sequence_depth(
    preds: Dict[str, Any],
    gt_paths: List[str],
    masks: Optional[List[str]],
    depth_cfg: Dict[str, Any],
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    逐序列深度评估（AbsRel / RMSE / δ）。
    会保存 per-frame CSV 与 per-seq JSON（若 out_dir 提供）。
    """
    pred_depths = _depths_from_outputs(preds)
    # print("perform depth evaluation",pred_depths)
    if pred_depths is None:
        return {"abs_rel": np.nan, "rmse": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}

    align = str(depth_cfg.get("align", "median"))
    min_d = float(depth_cfg.get("min_depth", 1e-3))
    max_d = float(depth_cfg.get("max_depth", 80.0))
    
    # print("gt_path: ", gt_paths)
    import sys
    np.set_printoptions(threshold=sys.maxsize, linewidth=10**9)
    # print("mask: ", masks)

    rows, per_frame = [], []
    for i, pd in enumerate(pred_depths):
        if i >= len(gt_paths): break
        gd = _read_depth(gt_paths[i])
        mk = _read_mask(masks[i]) if masks and i < len(masks) else None
        # print("gd:", gd)
        # print("pd:", mk)
        
        m = _depth_metrics_single(pd, gd, mk, align=align, min_depth=min_d, max_depth=max_d)
        per_frame.append(m); rows.append({"frame": i, **m})


    if out_dir is not None and rows:
        _save_csv(rows, Path(out_dir) / "depth_perframe.csv")

    if not per_frame:
        seq_mean = {"abs_rel": np.nan, "rmse": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}
    else:
        keys = per_frame[0].keys()
        seq_mean = {k: float(np.nanmean([m[k] for m in per_frame])) for k in keys}

    if out_dir is not None:
        _save_json({"sequence_mean": seq_mean, "per_frame": per_frame}, Path(out_dir) / "metrics_depth.json")
    return seq_mean


def aggregate_depth_metrics(per_seq_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_seq_metrics: return {}
    keys = per_seq_metrics[0].keys()
    return {k: float(np.nanmean([m.get(k, np.nan) for m in per_seq_metrics])) for k in keys}
