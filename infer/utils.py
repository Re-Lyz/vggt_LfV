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


def load_config(path: PathLike) -> dict:
    """
    读取 YAML 配置文件，返回 dict。
    """
    if yaml is None:
        raise ImportError("PyYAML 未安装。请先 `pip install pyyaml`。")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_config(cfg: dict, out_dir: PathLike, filename: str = "config_snapshot.yaml") -> None:
    """
    将配置字典写入到 out_dir/filename（YAML 格式），方便复现实验。
    """
    if yaml is None:
        raise ImportError("PyYAML 未安装。请先 `pip install pyyaml`。")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int = 42) -> None:
    """
    设定常见库的随机种子：random / numpy / torch（若可用）。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


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
        

def build_chunks(total_T: int, chunk_size: int) -> list[tuple[int, int]]:
    """返回 [(start, end), ...)，end 为开区间"""
    chunks = []
    s = 0
    while s < total_T:
        e = min(s + chunk_size, total_T)
        chunks.append((s, e))
        s = e
    return chunks

def is_dist():
    return dist.is_available() and dist.is_initialized()
def rank():
    return dist.get_rank() if is_dist() else 0
def world():
    return dist.get_world_size() if is_dist() else 1

def build_chunks(total_T: int, chunk_size: int):
    """把长度为 T 的序列按 chunk_size 切块，返回 [(s,e), ...)，e 为开区间"""
    if chunk_size <= 0 or chunk_size >= total_T:
        return [(0, total_T)]
    out, s = [], 0
    while s < total_T:
        e = min(s + chunk_size, total_T)
        out.append((s, e))
        s = e
    return out

def shard_chunks_for_rank(chunks, rank, world):
    """round-robin 分发 chunk 给每个 rank"""
    if world <= 1:
        return chunks
    return [ck for i, ck in enumerate(chunks) if i % world == rank]

def slice_align(x, s, e):
    """对 batch 中的对齐键做同样的 (s:e) 切片。支持 Tensor/list/tuple/None。"""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return x[s:e]
    if torch.is_tensor(x):
        return x[s:e]
    # 其它类型（标量/字典等）视为“不随帧维变化”的 metadata，直接原样带过去
    return x

def merge_preds_chunks(chunks_preds_list):
    """
    输入：[(s,e,preds_dict), ...]（已涵盖所有 rank）
    输出：按时间维拼接后的 merged_preds（dict / list / tensor）
    规则：逐键拼接；能 torch.cat 则 cat；不能 cat 就延展为 list 保序。
    """
    import torch
    # 展平 + 按 s 排序
    flat = []
    for part in chunks_preds_list:
        if not part:
            continue
        flat.extend(part)
    flat.sort(key=lambda t: t[0])  # 以 start 排序
    if not flat:
        return {}
    # 以第一个块的键为准，逐键收集
    merged = {}
    for _, _, pd in flat:
        assert isinstance(pd, dict), "Model should return a dict-like preds for stable merging."
        for k, v in pd.items():
            merged.setdefault(k, [])
            merged[k].append(v)
    # 逐键尝试 cat
    for k, arr in merged.items():
        try:
            # 尝试把 list 元素先规范为张量（有的可能是 list of tensors / numpy 等）
            if isinstance(arr[0], torch.Tensor):
                merged[k] = torch.cat(arr, dim=0)
            else:
                # 允许嵌套 list：把所有块的列表延展（如逐帧结构）
                out_list = []
                for item in arr:
                    if isinstance(item, (list, tuple)):
                        out_list.extend(list(item))
                    else:
                        out_list.append(item)
                merged[k] = out_list
        except Exception:
            # 不可 cat 的场景，退化为有序 list 延展
            out_list = []
            for item in arr:
                if isinstance(item, (list, tuple)):
                    out_list.extend(list(item))
                else:
                    out_list.append(item)
            merged[k] = out_list
    return merged

def shard_chunks_for_rank(chunks: list[tuple[int, int]], rank: int, world_size: int):
    """round-robin 给每个 rank 分到不重叠的 chunk 列表"""
    return [ck for i, ck in enumerate(chunks) if i % world_size == rank]

def dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if dist_is_ready() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist_is_ready() else 1

def setup_ddp(backend: str = "nccl", timeout_minutes: int = 60):
    """
    依赖 torchrun 注入的环境变量：RANK / WORLD_SIZE / LOCAL_RANK
    """
    if not dist.is_available() or dist_is_ready():
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=timeout_minutes),  # ← 用 datetime.timedelta
        rank=rank,
        world_size=world_size,
    )

def cleanup_ddp():
    if dist_is_ready():
        dist.destroy_process_group()

