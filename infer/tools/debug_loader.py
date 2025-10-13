# tools/debug_loader.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os
import cv2
import numpy as np
import torch


def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _to_numpy_image_grid(images_hw3: List[np.ndarray], cols: int = 4) -> np.ndarray:
    """把一批 HxWx3 的uint8图拼成九宫格。"""
    if not images_hw3:
        return np.zeros((10,10,3), dtype=np.uint8)
    h, w, _ = images_hw3[0].shape
    rows = int(np.ceil(len(images_hw3) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(images_hw3):
        r, c = divmod(i, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im
    return canvas

def _tensor_bchw_to_uint8_list(x: torch.Tensor) -> List[np.ndarray]:
    """
    x: [B,3,H,W] in [0,1] or similar -> List[H,W,3] uint8 (RGB)
    若不在[0,1]会做裁剪。
    """
    x = x.detach().float().cpu()
    B, C, H, W = x.shape
    imgs = []
    for i in range(B):
        xi = x[i]
        xi = torch.clamp(xi, 0.0, 1.0)
        im = (xi.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)  # HWC RGB
        imgs.append(im)
    return imgs

def _read_first_n(paths: List[str], n: int) -> List[np.ndarray]:
    outs = []
    for p in paths[:n]:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        outs.append(im)
    return outs

def _video_write_rgb(path: Path, frames_rgb: List[np.ndarray], fps: int = 10):
    if not frames_rgb: return
    h, w, _ = frames_rgb[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

def _exists_stats(paths: List[str]) -> Dict[str,int]:
    ex = sum(1 for p in paths if Path(p).is_file())
    return {"total": len(paths), "exists": ex, "missing": len(paths)-ex}

def _print_tensor_stats(name: str, t: torch.Tensor):
    if t is None:
        print(f"  {name}: None")
        return
    with torch.no_grad():
        tf = t.detach().float().to("cpu")
        shape = list(tf.shape)
        dtype = str(t.dtype)
        device = str(t.device)
        nan_count = int(torch.isnan(tf).sum().item())
        inf_count = int(torch.isinf(tf).sum().item())

        finite_mask = torch.isfinite(tf)
        if finite_mask.any():
            vals = tf[finite_mask]
            min_v = float(vals.min().item())
            max_v = float(vals.max().item())
        else:
            min_v = None
            max_v = None

        stats = {
            "shape": shape,
            "dtype": dtype,
            "device": device,
            "min_finite": min_v,
            "max_finite": max_v,
            "nan": nan_count,
            "inf": inf_count,
        }
    print(f"  {name}: {stats}")



def inspect_loader(
    loader,              # 你的 Adapter (get_dataset_loader 返回对象)
    device: str,
    out_dir: Path,
    max_sequences: int = 1,
    max_frames_visual: int = 12,
    fps: int = 8,
):
    """
    打印 & 可视化数据加载结果，确保模型输入正确：
      1) 打印序列列表数量、每序列的 images/gt/meta 概览
      2) 存原始前 N 帧九宫格/视频
      3) 用 loader.load_and_preprocess_images 做预处理，打印张量统计并可视化送模前的帧
      4) 对 gt（深度/掩码/法线路径）做存在性统计
    """
    _ensure_dir(out_dir)
    seq_ids: List[str] = loader.list_sequences()
    print(f"[debug_loader] num_sequences = {len(seq_ids)}")
    json_summary = {
        "num_sequences": len(seq_ids),
        "sequences": []
    }

    nseq = min(max_sequences, len(seq_ids))
    for sidx in range(nseq):
        sid = seq_ids[sidx]
        print(f"\n[debug_loader] === sequence[{sidx}] id: {sid} ===")
        seq_item = loader.build_sequence(sid)  # 需有 .images/.gt/.meta
        seq_out = out_dir / f"seq_{sidx:03d}"
        _ensure_dir(seq_out)

        # 1) 打印基本信息
        seq_info = {
            "sequence_name": seq_item.meta.get("sequence_name", f"seq_{sidx:03d}"),
            "num_images": len(seq_item.images),
            "first_images": seq_item.images[:max_frames_visual],
            "gt_keys": list(seq_item.gt.keys()),
            "meta_keys": list(seq_item.meta.keys()),
        }
        print(f"  num_images={seq_info['num_images']}")
        print(f"  first_images[0..{max_frames_visual}]:")
        for p in seq_info["first_images"]:
            print(f"    - {p}")
        print(f"  gt_keys={seq_info['gt_keys']}")
        print(f"  meta_keys={seq_info['meta_keys']}")

        # 2) 原图九宫格 / 小视频
        raw_frames = _read_first_n(seq_item.images, max_frames_visual)
        if raw_frames:
            grid = _to_numpy_image_grid(raw_frames, cols=4)
            cv2.imwrite(str(seq_out / "raw_grid.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            _video_write_rgb(seq_out / "raw_preview.mp4", raw_frames, fps=fps)
            print(f"  saved: {seq_out/'raw_grid.jpg'}, {seq_out/'raw_preview.mp4'}")
        else:
            print("  [warn] failed to read any raw frames for grid/video.")

        # 3) 预处理 → 模型输入张量
        try:
            images_tensor = loader.load_and_preprocess_images(seq_item.images[:max_frames_visual], device=device)
        except TypeError:
            # 某些实现只接受路径列表（不含 device）
            images_tensor = loader.load_and_preprocess_images(seq_item.images[:max_frames_visual]).to(device)

        if images_tensor.ndim == 4:   # [S,3,H,W]
            bchw = images_tensor
        elif images_tensor.ndim == 5: # [B,S,3,H,W] -> 取 batch 0
            bchw = images_tensor[0]
        else:
            raise RuntimeError(f"Unexpected tensor shape from loader.load_and_preprocess_images: {list(images_tensor.shape)}")

        print("  === preprocessed tensor ===")
        _print_tensor_stats("images_tensor", images_tensor)
        _print_tensor_stats("bchw_for_model", bchw)

        # 可视化预处理后的帧（裁剪到 [0,1] 再转 uint8）
        vis_list = _tensor_bchw_to_uint8_list(bchw)
        if vis_list:
            grid2 = _to_numpy_image_grid(vis_list, cols=4)
            cv2.imwrite(str(seq_out / "preprocessed_grid.jpg"), cv2.cvtColor(grid2, cv2.COLOR_RGB2BGR))
            _video_write_rgb(seq_out / "preprocessed_preview.mp4", vis_list, fps=fps)
            print(f"  saved: {seq_out/'preprocessed_grid.jpg'}, {seq_out/'preprocessed_preview.mp4'}")

        # 4) GT 存在性统计（有就统计）
        gt_stats = {}
        for key in ("depth_paths", "valid_masks", "normals_paths"):
            paths = seq_item.gt.get(key, None)
            if isinstance(paths, (list, tuple)):
                gt_stats[key] = _exists_stats(paths)
        print(f"  gt_exists: {gt_stats}")

        # 保存一个轻量 JSON
        seq_info["gt_exists"] = gt_stats
        json_summary["sequences"].append(seq_info)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[debug_loader] done. artifacts in: {out_dir}")
