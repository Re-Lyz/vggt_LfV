#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference backbone.

- 读取 YAML 配置
- 设定随机种子 / 设备与 AMP 策略
- 构建模型
- 构建数据加载器
- mode=inference: 由数据集完成 load_and_preprocess_images -> 推理 -> 评估 -> 可视化(占位)

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from tqdm import tqdm
import torch
from collections import defaultdict
import torch.nn as nn
import os
import torch
import torch.distributed as dist
import contextlib
from datetime import timedelta

# utils：配置 / IO / 随机种子 / 基础IO工具
from utils import (
    load_config,           # (cfg_path:str|Path) -> dict
    dump_config,           # (cfg:dict, out_dir:str|Path, filename:str="config_snapshot.yaml") -> None
    set_seed,              # (seed:int=42) -> None
    ensure_dir,            # (path:str|Path) -> None
    dump_json,             # (obj:Any, path:str|Path) -> None
)

# models：设备/精度策略与模型封装（占位）+ 微调相关占位
from models.vggt_wrapper import (
    pick_device,           # (device_cfg:str) -> (device:str, amp_dtype:torch.dtype)
    VGGTWrapper,           # class; VGGTWrapper(device, amp_dtype, amp_enabled)
)

# data：数据加载器工厂（推理 & 训练的占位接口）
from data.dataloader import (
    get_dataset_loader,      # (cfg:dict) -> obj: 需实现 .list_sequences() / .build_sequence(seq_id) / .load_and_preprocess_images(paths, device)

)

from tools.debug_loader import inspect_loader  # (loader, device, out_dir, max_sequences, max_frames_visual, fps) -> None

# eval：评估指标与聚合（推理用的占位）
from eval.evo_pose import evaluate_and_visualize_poses
from eval.depth import evaluate_sequence_depth, aggregate_depth_metrics

# vis：可视化（占位）
# from vis.camera_path import (
#     extract_cam_centers,       # (predictions:Dict[str,Any]) -> Optional[np.ndarray[B,3]]
#     save_3d_path_plot,         # (centers:np.ndarray[B,3], out_path:Path) -> None
# )
# from vis.video_overlay import (
#     make_video_with_overlays,  # (image_paths:List[str], out_path:Path, traj_normed_xy:Optional[np.ndarray[B,2]], fps:int, draw_traj:bool, draw_cam_axis:bool, axis_len:int) -> None
# )

# === 新增：适配器，把 VGGTWrapper 暴露为 nn.Module 的 forward ===
class _DPInferAdapter(nn.Module):
    def __init__(self, wrapper: VGGTWrapper):
        super().__init__()
        self.wrapper = wrapper  # 不展开到子模块，避免 state_dict 名称变化

    def forward(self, images: torch.Tensor, query_points: Optional[torch.Tensor] = None):
        # 直接调用你现有的单卡推理入口
        return self.wrapper.infer_images(images, query_points=query_points)



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


# -----------------------
# 公共：一次前向（推理）
# -----------------------
def _forward_infer(
    model: nn.Module,
    images: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    query_points: Optional[torch.Tensor] = None,
):
    with torch.no_grad():
        use_amp = (images.is_cuda and amp_enabled)
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        with ctx:
            # 这里 model 是 DDP 包裹的 _DPInferAdapter，直接调用即可
            return model(images, query_points)



# -----------------------
# 推理主流程
# -----------------------
def run_inference(cfg, device: str, amp_dtype: torch.dtype, amp_enabled: bool, args=None):
    out_root = Path(cfg.get("output_dir", "outputs/infer"))
    ensure_dir(out_root)
    dump_config(cfg, out_root)

    # ===== 配置并初始化 DDP =====
    ddp_conf = cfg.get("inference", {})
    use_ddp = bool(ddp_conf.get("ddp", False))
    only_rank0_visual = bool(ddp_conf.get("only_rank0_visual", True))
    use_rank_subdir = bool(ddp_conf.get("rank_output_subdir", True))
    infer_cap = int(ddp_conf.get("max_img_per_gpu", -1))

    if use_ddp and torch.cuda.is_available():
        setup_ddp(
            backend=ddp_conf.get("ddp_backend", "nccl"),
            timeout_minutes=int(ddp_conf.get("ddp_timeout_mins", 60)),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
    else:
        # 单卡路径（不初始化进程组）
        device = "cuda:0" if (device == "auto" and torch.cuda.is_available()) else device

    # ===== 构建模型 =====
    # 包一层适配器以便 DDP 正常 forward（适配器内部调用 wrapper.infer_images）
    wrapper = VGGTWrapper(device=device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
    model = _DPInferAdapter(wrapper).to(device)

    if use_ddp and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[int(device.split(":")[1])] if device.startswith("cuda") else None,
            output_device=int(device.split(":")[1]) if device.startswith("cuda") else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # ===== 数据集 & 序列划分（DDP 每个 rank 处理不重叠子集）=====
    loader = get_dataset_loader(cfg)
    seq_ids = loader.list_sequences()
    limit = cfg.get("dataset", {}).get("limit_seqs", -1)
    if isinstance(limit, int) and limit > 0:
        seq_ids = seq_ids[:limit]

    rank = get_rank()
    world_size = get_world_size()
    if use_ddp and world_size > 1:
        seq_ids = seq_ids[rank::world_size]  # 轮转切分，避免重复

    # ===== 推理主循环 =====
    all_seq_depth_metrics = []
    pose_collect = defaultdict(lambda: {"APE": defaultdict(list), "RPE": defaultdict(list)})

    iterator = tqdm(seq_ids, desc="Inference") if rank == 0 else seq_ids
    for seq_id in iterator:
        try:
            seq_item = loader.build_sequence(seq_id)
            seq_out = out_root / seq_item.meta.get("sequence_name", f"seq_{seq_id}")

            # rank 输出目录策略
            if rank == 0 or not use_rank_subdir:
                out_dir_this_root = seq_out
            else:
                out_dir_this_root = seq_out / f"rank{rank}"
            ensure_dir(out_dir_this_root)

            # 读取并预处理图像到当前设备
            images = loader.load_and_preprocess_images(seq_item.images, device=device)

            # 可选：限制每次前向的帧上限（以帧计的 minibatch cap）
            if isinstance(images, torch.Tensor) and images.dim() >= 1 and infer_cap > 0:
                if images.shape[0] > infer_cap:
                    images = images[:infer_cap]

            # （可选）查询点
            query_points = None
            if hasattr(loader, "load_and_preprocess_query_points"):
                try:
                    query_points = loader.load_and_preprocess_query_points(seq_item, device=device)
                except Exception:
                    query_points = None

            # === 前向推理 ===
            preds = _forward_infer(
                model=model,
                images=images,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                query_points=query_points,
            )

            # === 位姿评估 ===
            cam_cfg = cfg.get("evaluation", {}).get("camera", {})
            if cam_cfg.get("enabled", True):
                cam_cfg_eff = dict(cam_cfg)
                # 非 rank0 禁用绘图，避免并发可视化冲突
                if only_rank0_visual and rank != 0:
                    cam_cfg_eff["plot_3d"] = False

                pose_results_by_mode = evaluate_and_visualize_poses(
                    preds=preds,
                    seq_item=seq_item,
                    out_dir=out_dir_this_root,
                    camera_cfg=cam_cfg_eff,
                )
                for mode, res in pose_results_by_mode.items():
                    for mname, mdict in res.items():
                        for k, v in mdict.items():
                            if isinstance(v, (int, float)):
                                pose_collect[mode][mname][k].append(v)

            # === 深度评估 ===
            depth_cfg = cfg.get("evaluation", {}).get("depth", {"enabled": True})
            if depth_cfg.get("enabled", True):
                depth_cfg_eff = dict(depth_cfg)
                # 如需也在非 rank0 关闭可视化，可在这里调整 depth_cfg_eff 的相关开关
                depth_metrics = evaluate_sequence_depth(
                    preds=preds,
                    gt_paths=seq_item.gt.get("depth_paths", []),
                    masks=seq_item.gt.get("valid_masks", None),
                    depth_cfg=depth_cfg_eff,
                    out_dir=out_dir_this_root,
                )
                all_seq_depth_metrics.append(depth_metrics)
            else:
                depth_metrics = {}

            # 每序列的 metrics 快速落盘
            dump_json({
                "sequence": seq_item.meta.get("sequence_name", str(seq_id)),
                "depth": depth_metrics,
            }, out_dir_this_root / "metrics_summary.json")

        except Exception as e:
            print(f"[WARN][rank {rank}] sequence {seq_id} failed: {e}")

    # ===== 跨 rank 汇总到 rank0 =====
    if use_ddp and world_size > 1:
        gather_list = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(
            obj=dict(depth=all_seq_depth_metrics, pose=pose_collect),
            gather_list=gather_list,
            dst=0,
        )

        if rank == 0:
            merged_depth = []
            merged_pose = defaultdict(lambda: {"APE": defaultdict(list), "RPE": defaultdict(list)})
            for it in gather_list:
                merged_depth.extend(it["depth"])
                for mode, groups in it["pose"].items():
                    for mname, stats in groups.items():
                        for k, arr in stats.items():
                            merged_pose[mode][mname][k].extend(arr)
            all_seq_depth_metrics = merged_depth
            pose_collect = merged_pose

    # ===== 只有 rank0 写最终汇总 =====
    if rank == 0:
        summary = {}
        if all_seq_depth_metrics:
            summary["depth_overall_mean"] = aggregate_depth_metrics(all_seq_depth_metrics)

        pose_overall = {}
        for mode, groups in pose_collect.items():
            pose_overall[mode] = {}
            for mname, stats_dict in groups.items():
                vals = {k: float(sum(v) / len(v)) for k, v in stats_dict.items() if len(v) > 0}
                if vals:
                    pose_overall[mname] = vals
        if pose_overall:
            summary["camera_overall_mean"] = pose_overall

        dump_json({"summary": summary, "num_sequences": len(loader.list_sequences())}, out_root / "summary.json")
        print("[rank0] Inference done. Summary:", json.dumps(summary, indent=2, ensure_ascii=False))

    if use_ddp and dist_is_ready():
        cleanup_ddp()

# def test(cfg, device: str, amp_dtype: torch.dtype, amp_enabled: bool, args=None):
#     out_root = Path(cfg.get("output_dir", "outputs/infer"))
#     ensure_dir(out_root)
#     dump_config(cfg, out_root)

#     # 模型
#     model = VGGTWrapper(device=device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)

#     # 数据集（要求 loader 自带 load_and_preprocess_images）
#     loader = get_dataset_loader(cfg)
#     data1 = loader.base.get_data(seq_index=0)
#     data2 = loader.base.get_data(seq_index=1)
    
#     print(data1["frame_num"])
#     print(data2["frame_num"])

# -----------------------
# 入口
# -----------------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT inference/finetune runner")
    parser.add_argument(
        "-c", "--config",
        default="config/default.yaml",   # 默认配置文件路径
        help="Path to the YAML config file"
    )
    parser.add_argument("--debug-loader", action="store_true", help="可视化/检查 loader 输出并退出")
    parser.add_argument("--debug-max-seqs", type=int, default=1)
    parser.add_argument("--debug-max-frames", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    cfg["_cfg_dir"] = str(cfg_path.parent.resolve())
    set_seed(123)

    # 设备/AMP
    device_cfg = cfg.get("device", "auto")
    amp_cfg = cfg.get("amp", {"enabled": True, "dtype_policy": "auto"})
    device, auto_amp_dtype = pick_device(device_cfg)

    dtype_policy = amp_cfg.get("dtype_policy", "auto")
    if dtype_policy == "auto":
        amp_dtype = auto_amp_dtype
    elif dtype_policy == "bfloat16":
        amp_dtype = torch.bfloat16
    elif dtype_policy == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    amp_enabled = bool(amp_cfg.get("enabled", True))

    mode = str(cfg.get("mode", "inference")).lower()
    if mode == "inference":
        run_inference(cfg, device, amp_dtype, amp_enabled, args=args)
        # test(cfg, device, amp_dtype, amp_enabled, args=args)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

if __name__ == "__main__":
    main()
