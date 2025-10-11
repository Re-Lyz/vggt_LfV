# data/dataloader.py
from __future__ import annotations
from pathlib import Path
import inspect
import types
import yaml
from data.datasets.c3vd import C3VDDatasetV1Adapter, C3VDDatasetv1  
from types import SimpleNamespace

def _to_ns(obj):
    """递归地把 dict/list 转成 SimpleNamespace / list[SimpleNamespace]。"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj

def _load_dataset_section(cfg: dict) -> dict:
    ds = cfg.get("dataset", {})
    cfg_dir = Path(cfg.get("_cfg_dir", "."))  # infer.py 里记得设置
    sub = ds.get("config_file", None)
    if sub is None:
        raise ValueError("dataset.config_file 未设置（例如 custom_dataset.yaml）")
    
    cfg_dir = (
        Path(cfg.get("_cfg_dir")).resolve()
        if cfg.get("_cfg_dir")
        else Path(cfg.get("_config_path", "")).resolve().parent
        if cfg.get("_config_path")
        else Path.cwd()  # 最后兜底；正常不会走到
    )
    
    ds_yaml_path = (cfg_dir / sub).resolve()
    print(f"Loading dataset config from: {ds_yaml_path}")
    if not ds_yaml_path.is_file():
        raise FileNotFoundError(f"dataset config not found: {ds_yaml_path}")
    with open(ds_yaml_path, "r", encoding="utf-8") as f:
        ds_yaml = yaml.safe_load(f) or {}
    split = "val" if str(cfg.get("mode", "inference")).lower() == "inference" else "train"
    return (ds_yaml.get("data") or {}).get(split, {})

def get_dataset_loader(cfg: dict):
    ds = cfg.get("dataset", {})
    root = ds.get("root")
    if not root:
        raise ValueError("数据集路径未设置")

    section = _load_dataset_section(cfg)
    common_cfg_dict = section.get("common_config", {})
    # SimpleNamespace：把 dict 变成能用属性访问的对象，满足 C3VDDatasetv1 的 common_conf 需求
    common_conf = _to_ns(common_cfg_dict) 

    # 组装 C3VDDatasetv1 构造参数
    split = "train" if str(cfg.get("mode", "inference")).lower() == "finetune" else "val"
    ctor_kwargs = dict(
        common_conf=common_conf,
        split=("train" if split == "train" else "test" if section.get("common_config", {}).get("training", False) is False else "train"),
        ROOT=root,
        USE_REGISTERED=bool(ds.get("use_registered", True)),
        min_num_images=int(ds.get("min_num_images", 2)),
        assume_pose=str(ds.get("assume_pose", "w2c")),
        depth_unit_scale=float(ds.get("depth_unit_scale", 1.0)),
    )

    # 实例化 base
    base = C3VDDatasetv1(**{k: v for k, v in ctor_kwargs.items() if k in inspect.signature(C3VDDatasetv1).parameters})

    # 用 Adapter 包一层，返回给 infer
    return C3VDDatasetV1Adapter(base=base, ds_cfg=ds)

def get_train_dataloader(cfg: dict):
    raise NotImplementedError("get_train_dataloader is not implemented yet.")

def get_eval_dataloader(cfg: dict):
    raise NotImplementedError("get_eval_dataloader is not implemented yet.")
