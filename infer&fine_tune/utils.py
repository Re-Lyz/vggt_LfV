# utils.py
from __future__ import annotations
import os
import json
import random
from pathlib import Path
from typing import Any, Union

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
