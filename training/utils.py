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
        
        
        
        
        
        
        
        
        

