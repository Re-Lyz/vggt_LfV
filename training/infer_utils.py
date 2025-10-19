# utils.py
from __future__ import annotations
import os
import json
import random
from pathlib import Path
from typing import Any, Union
import torch
import torch.distributed as dist
from datetime import timedelta

# 可选依赖：yaml（用于读取/写入配置）
try:
    import yaml
except Exception as e:
    yaml = None

# 可选依赖：numpy / torch（用于设定随机种子，缺失时自动降级）
try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


PathLike = Union[str, Path]



def ensure_dir(path: PathLike) -> None:
    """
    确保目录存在（不存在则递归创建）。
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def dump_json(obj: Any, path: PathLike) -> None:
    """
    将对象以 JSON 写入到 path（会自动创建父目录）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
         
_DTYPE_CAST = {
    torch.bfloat16: torch.float32,  # numpy 不支持 bfloat16，先转 float32
    torch.float16: torch.float32,   # 有些下游算子对 fp16 不稳
}
      
def tensor_to_numpy(t) :
    dt = _DTYPE_CAST.get(t.dtype, None)
    if dt is not None:
        t = t.to(dt)
    return t.detach().cpu().numpy()

def tree_to_numpy(x):
    """仅把 torch.Tensor→numpy；保持原有 dict/list/tuple 结构；其他类型原样返回。"""
    if torch.is_tensor(x):
        return tensor_to_numpy(x)
    if isinstance(x, dict):
        return {k: tree_to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out = [tree_to_numpy(v) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    if isinstance(x, (np.ndarray, np.number, float, int, str)) or x is None:
        return x
    # 其他罕见类型（Path、bool、自定义类等）直接原样返回
    return x
        
def as_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, device=device)

def sort_batch_by_ids(batch: dict):
    """
    用 batch['ids'] 对 batch 中所有含时间维的条目做一致排序（升序）。
    支持 ids 为 [B,T] 或 [T]。若缺失，原样返回。
    """
    if "ids" not in batch or batch["ids"] is None:
        return batch

    # 取设备；优先 images 的设备
    dev = None
    for k in ("images", "depths", "intrinsics", "extrinsics"):
        if k in batch and isinstance(batch[k], torch.Tensor):
            dev = batch[k].device
            break
    if dev is None:
        dev = torch.device("cpu")

    ids = as_tensor(batch["ids"], dev)
    # 统一成 (B,T)
    if ids.ndim == 1:
        # [T]，对所有 B 共享同一顺序
        T = ids.shape[0]
        order_per_b = ids.argsort().unsqueeze(0)   # [1,T]
        shared = True
    elif ids.ndim == 2:
        # [B,T]，每个样本各自排序
        order_per_b = ids.argsort(dim=1)           # [B,T]
        shared = False
        T = ids.shape[1]
    else:
        # 其他形状不支持，直接返回
        return batch

    # 推断 B
    B = None
    if "images" in batch and isinstance(batch["images"], torch.Tensor):
        if batch["images"].ndim == 5:  # [B,T,C,H,W]
            B = batch["images"].shape[0]
        elif batch["images"].ndim == 4:  # [B,C,H,W] → 视为 T=1，无需重排
            B = batch["images"].shape[0]
    if B is None and ids.ndim == 2:
        B = ids.shape[0]

    def _reorder_tensor(x):
        # 仅重排含有时间维的张量
        if x.ndim >= 2 and B is not None and x.shape[0] == B and x.shape[1] == T:
            if shared:
                return x[:, order_per_b[0], ...]
            else:
                # 每个样本单独排序（安全但易懂）
                return torch.stack([x[b, order_per_b[b], ...] for b in range(B)], dim=0)
        if x.ndim >= 1 and not (B is None) and x.shape[0] == T and (ids.ndim == 1):
            # [T,...] 的情况（无 batch 维）
            return x[order_per_b[0], ...]
        return x  # 形状不匹配则不动

    def _reorder_np(x):
        if x.ndim >= 2 and B is not None and x.shape[0] == B and x.shape[1] == T:
            if shared:
                return x[:, order_per_b[0].cpu().numpy(), ...]
            else:
                return np.stack([x[b, order_per_b[b].cpu().numpy(), ...] for b in range(B)], axis=0)
        if x.ndim >= 1 and (ids.ndim == 1) and x.shape[0] == T:
            return x[order_per_b[0].cpu().numpy(), ...]
        return x

    # 按键遍历并重排
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor):
            batch[key] = _reorder_tensor(val)
        elif isinstance(val, np.ndarray):
            batch[key] = _reorder_np(val)
        elif isinstance(val, list):
            # 假如有 [B,T] 的 list（少见），做个简单支持
            try:
                arr = np.array(val, dtype=object)
                batch[key] = _reorder_np(arr).tolist()
            except Exception:
                pass

    # ids 本身也重排成升序（方便后续可视化/日志）
    if ids.ndim == 1:
        batch["ids"] = ids[order_per_b[0]].to(ids.dtype)
    else:
        batch["ids"] = torch.stack([ids[b, order_per_b[b]] for b in range(ids.shape[0])], dim=0).to(ids.dtype)

    return batch        
        
        
def merge_camera_results(all_res: list[dict]) -> dict:
    if not all_res:
        return {}
    # 允许聚合的数值键（按你项目里真实产出补全/调整）
    AGG_KEYS_APE = {"trans_rmse", "trans_mean", "trans_median", "trans_std",
                    "rot_rmse_deg", "rot_mean_deg", "rot_median_deg", "rot_std_deg"}
    AGG_KEYS_RPE = {"trans_rmse", "trans_mean", "trans_median", "trans_std",
                    "rot_rmse_deg", "rot_mean_deg", "rot_median_deg", "rot_std_deg"}
    # 特殊处理键
    SUM_KEYS = {"num_poses"}         # 计数型求和
    PASS_META = {"mode", "delta", "delta_unit"}  # 直接透传（取首个非空）

    modes = {}
    for res in all_res:
        for mode, block in res.items():
            m = modes.setdefault(mode, {"APE": [], "RPE": [], "META": []})
            if "APE" in block: m["APE"].append(block["APE"])
            if "RPE" in block: m["RPE"].append(block["RPE"])
            # 收集可选元数据（可能在 APE/RPE 内部，也可能在顶层）
            meta = {}
            for key in PASS_META:
                # 在 APE/RPE 顶层查找，再到子块里兜底
                meta[key] = block.get(key, None)
                if meta[key] is None:
                    meta[key] = block.get("APE", {}).get(key, None)
                if meta[key] is None:
                    meta[key] = block.get("RPE", {}).get(key, None)
            m["META"].append(meta)

    merged = {}
    for mode, parts in modes.items():
        merged[mode] = {}

        def _agg_block(items, allowed_keys):
            out = {}
            # 先处理 SUM_KEYS
            for k in SUM_KEYS:
                vals = []
                for d in items:
                    v = d.get(k, None)
                    if v is None: continue
                    try: vals.append(float(v))
                    except: pass
                out[k] = int(sum(vals)) if vals else 0
            # 再处理均值键
            for k in allowed_keys:
                vals = []
                for d in items:
                    v = d.get(k, None)
                    if v is None: continue
                    try: vals.append(float(v))
                    except: pass
                if vals:
                    out[k] = float(np.mean(vals))
            return out

        merged[mode]["APE"] = _agg_block(parts["APE"], AGG_KEYS_APE)
        merged[mode]["RPE"] = _agg_block(parts["RPE"], AGG_KEYS_RPE)

        # 透传元数据（取第一个非空）
        meta_out = {}
        for key in PASS_META:
            first = next((m.get(key) for m in parts["META"] if m.get(key) is not None), None)
            if first is not None:
                meta_out[key] = first
        if meta_out:
            merged[mode]["meta"] = meta_out

    return merged

def to_uint8_chw(img_t):
    """
    img_t: torch.Tensor[C,H,W] or np.ndarray[H,W,C]/[C,H,W]
    返回 torch.uint8[C,H,W]，范围 0..255
    """
    import torch, numpy as np
    if isinstance(img_t, np.ndarray):
        img_t = torch.from_numpy(img_t)

    if img_t.ndim == 3 and img_t.shape[0] in (1,3):  # [C,H,W]
        x = img_t
    elif img_t.ndim == 3 and img_t.shape[-1] in (1,3):  # [H,W,C]
        x = img_t.permute(2,0,1)
    else:
        raise RuntimeError(f"Unexpected image shape for save: {tuple(img_t.shape)}")

    x = x.detach().cpu()

    if x.dtype == torch.uint8:
        return x

    # float → uint8：自动判断范围（支持 [-1,1] / [0,1] / [0,255]）
    x = x.to(torch.float32)
    vmin, vmax = float(x.min()), float(x.max())
    if vmax <= 1.0 + 1e-3 and vmin >= -1.0 - 1e-3:
        # 可能是 [-1,1] 或 [0,1]
        if vmin < -0.05:
            x = (x * 0.5 + 0.5)  # [-1,1] → [0,1]
        x = (x.clamp(0,1) * 255.0).round()
    else:
        # 可能已经是 [0,255] 浮点
        x = x.clamp(0,255).round()
    return x.to(torch.uint8)


def save_batch_images(batch, out_dir: Path, rank: int):
    """
    将当前 batch 的图像保存到 out_dir/images 下。
    命名： {seq_name}__{id}.png
    兼容多样本（B>1）和多帧（T>1）。
    """
    import torch, numpy as np
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = batch.get("images", None)
    if images is None:
        print(f"[rank {rank}] no 'images' in batch; skip saving.")
        return

    # 取 seq_name（可能是 str / List[str]）
    seq_name = batch.get("seq_name", None)
    if isinstance(seq_name, str):
        seq_names = [seq_name]
    elif isinstance(seq_name, (list, tuple)):
        seq_names = list(seq_name)
    else:
        # 有些 collate 会把字符串打成 list-of-lists；兜底转成字符串
        seq_names = [str(seq_name)]

    # 取 ids（每个样本对应一组 frame ids）
    # 可能是 ndarray[List[int]] / List[np.ndarray] / Tensor 等
    ids_any = batch.get("ids", None)

    def _get_ids_for_sample(ids_any, b_idx):
        if ids_any is None:
            return None
        import torch, numpy as np
        try:
            if isinstance(ids_any, torch.Tensor):
                ids_b = ids_any[b_idx]
                ids_b = ids_b.detach().cpu().numpy()
            elif isinstance(ids_any, (list, tuple)):
                ids_b = ids_any[b_idx]
                if isinstance(ids_b, torch.Tensor):
                    ids_b = ids_b.detach().cpu().numpy()
                elif isinstance(ids_b, np.ndarray):
                    pass
                else:
                    ids_b = np.asarray(ids_b)
            else:
                ids_b = ids_any
                if isinstance(ids_b, torch.Tensor):
                    ids_b = ids_b.detach().cpu().numpy()
                elif not isinstance(ids_b, np.ndarray):
                    ids_b = np.asarray(ids_b)
            return ids_b
        except Exception:
            return None

    # 统一成 torch.Tensor
    import torch
    if isinstance(images, torch.Tensor):
        imgs = images
        # 支持 [B,T,C,H,W] / [B,C,H,W] / [T,C,H,W] / [C,H,W]
        if imgs.ndim == 5:
            B, T, C, H, W = imgs.shape
        elif imgs.ndim == 4:
            # 视作 [B,C,H,W]
            B, C, H, W = imgs.shape
            T = 1
            imgs = imgs.unsqueeze(1)  # → [B,1,C,H,W]
        elif imgs.ndim == 3:
            # 单样本单帧 → [1,1,C,H,W]
            C, H, W = imgs.shape
            B, T = 1, 1
            imgs = imgs.unsqueeze(0).unsqueeze(0)
        else:
            raise RuntimeError(f"Unsupported images tensor shape: {tuple(imgs.shape)}")

        for b in range(B):
            name_b = seq_names[b] if b < len(seq_names) else f"sample{b}"
            ids_b = _get_ids_for_sample(ids_any, b)
            for t in range(T):
                id_str = None
                if ids_b is not None:
                    try:
                        id_str = f"{int(ids_b[t])}"
                    except Exception:
                        pass
                if id_str is None:
                    id_str = f"{t:04d}"
                fn = f"{name_b}__{id_str}.png"
                x = to_uint8_chw(imgs[b, t])  # [C,H,W] uint8
                # 保存（RGB）
                from PIL import Image
                x_np = x.permute(1,2,0).numpy()  # [H,W,C]
                Image.fromarray(x_np).save(img_dir / fn)

    else:
        # list / ndarray 版本（例如 list of length B，元素是 [T,C,H,W] / [H,W,C]）
        import numpy as np
        arr = images
        if isinstance(arr, np.ndarray):
            # 可能是 [B,T,C,H,W] / [B,C,H,W] / ...
            if arr.ndim == 5:
                B, T = arr.shape[0], arr.shape[1]
            elif arr.ndim == 4:
                B, T = arr.shape[0], 1
                arr = arr[:,None,...]
            elif arr.ndim == 3:
                B, T = 1, 1
                arr = arr[None,None,...]
            else:
                raise RuntimeError(f"Unsupported ndarray images shape: {arr.shape}")
            for b in range(B):
                name_b = seq_names[b] if b < len(seq_names) else f"sample{b}"
                ids_b = _get_ids_for_sample(ids_any, b)
                for t in range(T):
                    id_str = None
                    if ids_b is not None:
                        try:
                            id_str = f"{int(ids_b[t])}"
                        except Exception:
                            pass
                    if id_str is None:
                        id_str = f"{t:04d}"
                    fn = f"{name_b}__{id_str}.png"
                    x = to_uint8_chw(arr[b,t])       # → [C,H,W] uint8
                    from PIL import Image
                    Image.fromarray(x.permute(1,2,0).numpy()).save(img_dir / fn)
        else:
            # list 风格
            B = len(arr)
            for b in range(B):
                name_b = seq_names[b] if b < len(seq_names) else f"sample{b}"
                ids_b = _get_ids_for_sample(ids_any, b)
                frames = arr[b]
                # frames 可能是 [T,C,H,W] / [H,W,C] / list
                if isinstance(frames, (list, tuple)):
                    T = len(frames)
                    for t in range(T):
                        id_str = None
                        if ids_b is not None:
                            try:
                                id_str = f"{int(ids_b[t])}"
                            except Exception:
                                pass
                        if id_str is None:
                            id_str = f"{t:04d}"
                        fn = f"{name_b}__{id_str}.png"
                        x = to_uint8_chw(frames[t])
                        from PIL import Image
                        Image.fromarray(x.permute(1,2,0).numpy()).save(img_dir / fn)
                else:
                    # 单帧
                    fn = f"{name_b}__0000.png"
                    x = to_uint8_chw(frames)
                    from PIL import Image
                    Image.fromarray(x.permute(1,2,0).numpy()).save(img_dir / fn)
        
        
        
        
        

