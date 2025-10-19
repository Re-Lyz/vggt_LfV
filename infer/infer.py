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
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os, json
from pathlib import Path
from collections import defaultdict

# utils：配置 / IO / 随机种子 / 基础IO工具
from utils import *

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

_DTYPE_CAST = {
    torch.bfloat16: torch.float32,  # numpy 不支持 bfloat16，先转 float32
    torch.float16: torch.float32,   # 有些下游算子对 fp16 不稳
}

def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    dt = _DTYPE_CAST.get(t.dtype, None)
    if dt is not None:
        t = t.to(dt)
    return t.detach().cpu().numpy()

def tree_to_numpy(x):
    """仅把 torch.Tensor→numpy；保持原有 dict/list/tuple 结构；其他类型原样返回。"""
    if torch.is_tensor(x):
        return _tensor_to_numpy(x)
    if isinstance(x, dict):
        return {k: tree_to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out = [tree_to_numpy(v) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    if isinstance(x, (np.ndarray, np.number, float, int, str)) or x is None:
        return x
    # 其他罕见类型（Path、bool、自定义类等）直接原样返回
    return x

# -----------------------
# 公共：一次前向（推理）
# -----------------------
def _forward_infer(
    model,
    images: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    query_points: Optional[torch.Tensor] = None,
    **extra,  # 预留：有需要时可透传 intrinsics/extrinsics/masks 等
):
    """
    单次前向：对子批 (s:e) 做 no_grad + 可选 autocast。
    - model 可以是 DDP 包裹的 _DPInferAdapter；
    - 不在这里 .to(device)，外层已搬到正确 device；
    - 返回值保持模型原样（通常为 dict）。
    """
    # 组装 forward 的参数
    fwd_kwargs = {"images": images}
    if query_points is not None:
        fwd_kwargs["query_points"] = query_points
    if extra:
        fwd_kwargs.update(extra)

    # 选择 autocast 的 device_type（以 images 为准，若无则看是否有 CUDA）
    if torch.is_tensor(images) and images.is_cuda:
        device_type = "cuda"
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # 前向（不做 grad、可选 AMP）
    if amp_enabled:
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=amp_dtype):
            preds = model(**fwd_kwargs)
    else:
        with torch.no_grad():
            preds = model(**fwd_kwargs)

    return preds



# -----------------------
# 推理主流程
# -----------------------
def run_inference(cfg, device: str, amp_dtype: torch.dtype, amp_enabled: bool, args=None):
    """
    分块-分发-并行前向-汇总拼接 的分布式推理版本：
    - 不再按序列在 rank 间轮转切分；
    - 同一条序列按 inference.max_img_per_gpu 切 chunk，round-robin 分给各 rank；
    - 各 rank 仅前向自己领取的 (s,e) 子段；
    - 用 dist.gather_object 把 (s,e,preds_chunk) 收拢到 rank0，按时间维拼回完整序列；
    - 评估与可视化、序列 metrics 与最终 summary 的逻辑保持不变，均在 rank0 执行。
    依赖的功能函数（请在文件其他位置或 utils 中提供）：
        build_chunks(total_T:int, chunk_size:int) -> list[(s,e)]
        shard_chunks_for_rank(chunks, rank:int, world_size:int) -> list[(s,e)]
        slice_align(x, s:int, e:int) -> 同步对齐切片
        merge_preds_chunks(gathered_per_rank: list[list[(s,e,preds_dict)]]) -> dict(按维度0拼接)
    """


    out_root = Path(cfg.get("output_dir", "outputs/infer"))
    ensure_dir(out_root)
    dump_config(cfg, out_root)

    # ===== 配置并初始化 DDP =====
    ddp_conf = cfg.get("inference", {})
    use_ddp = bool(ddp_conf.get("ddp", False))
    only_rank0_visual = bool(ddp_conf.get("only_rank0_visual", True))
    use_rank_subdir = bool(ddp_conf.get("rank_output_subdir", True))
    infer_cap = int(ddp_conf.get("max_img_per_gpu", -1))  # 现在语义：chunk_size；<=0 表示整段

    if use_ddp and torch.cuda.is_available():
        setup_ddp(
            backend=ddp_conf.get("ddp_backend", "nccl"),
            timeout_minutes=int(ddp_conf.get("ddp_timeout_mins", 60)),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
    else:
        device = "cuda:0" if (device == "auto" and torch.cuda.is_available()) else device

    # ===== 构建模型 =====
    wrapper = VGGTWrapper(device=device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
    model = _DPInferAdapter(wrapper).to(device)

    if use_ddp and torch.cuda.is_available():
        # 只有在模型里存在需要梯度的参数时才包 DDP；否则保持原样（仍可用 dist.gather/barrier）
        need_ddp_wrap = any(p.requires_grad for p in model.parameters())
        if need_ddp_wrap:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[int(device.split(":")[1])] if device.startswith("cuda") else None,
                output_device=int(device.split(":")[1]) if device.startswith("cuda") else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            if get_rank() == 0:
                print("[info] DDP wrap skipped: no parameter requires grad; using dist for coordination only.")

    # ===== 数据集（所有 rank 同步遍历相同序列）=====
    loader = get_dataset_loader(cfg)
    seq_ids = loader.list_sequences()
    limit = cfg.get("dataset", {}).get("limit_seqs", -1)
    if isinstance(limit, int) and limit > 0:
        seq_ids = seq_ids[:limit]

    rank = get_rank()
    world_size = get_world_size()

    # 关键改动：不再对 seq_ids 做 rank 级切分（seq_ids[rank::world_size]）
    iterator = tqdm(seq_ids, desc="Inference") if rank == 0 else seq_ids

    # ===== 指标收集容器（最终仍然 summary）=====
    all_seq_depth_metrics = []
    pose_collect = defaultdict(lambda: {"APE": defaultdict(list), "RPE": defaultdict(list)})

    for seq_id in iterator:
        try:
            seq_item = loader.build_sequence(seq_id)
            seq_name = seq_item.meta.get("sequence_name", f"seq_{seq_id}")
            seq_out = out_root / seq_name

            # rank 输出目录策略（评估只在 rank0 执行，但目录创建保持不变）
            if rank == 0 or not use_rank_subdir:
                out_dir_this_root = seq_out
            else:
                out_dir_this_root = seq_out / f"rank{rank}"
            ensure_dir(out_dir_this_root)

            # 读取并预处理全序列图像到当前设备（每个 rank 都加载；随后各自按 (s,e) 切片）
            images = loader.load_and_preprocess_images(seq_item.images, device=device)
            T = images.shape[0] if torch.is_tensor(images) else len(images)

            # 分块（<=0 表示整段作为一个块）
            chunk_size = infer_cap if (infer_cap and infer_cap > 0) else T
            chunks_all = build_chunks(T, chunk_size)                     # e.g. [(0,32),(32,64),...]
            chunks_my = shard_chunks_for_rank(chunks_all, rank, world_size)  # round-robin 分给当前 rank

            # （可选）查询点等“按帧对齐”的键
            query_points = None
            if hasattr(loader, "load_and_preprocess_query_points"):
                try:
                    query_points = loader.load_and_preprocess_query_points(seq_item, device=device)
                except Exception:
                    query_points = None

            # === 本 rank 仅推理自己领取到的 chunk ===
            local_results = []  # [(s, e, preds_chunk_dict), ...]
            for (s, e) in chunks_my:
                sub_images = slice_align(images, s, e)
                sub_qpts = slice_align(query_points, s, e)

                preds_chunk = _forward_infer(
                    model=model,
                    images=sub_images,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    query_points=sub_qpts,
                )
                if isinstance(preds_chunk, dict):
                    local_results.append((s, e, preds_chunk))
                else:
                    local_results.append((s, e, {"_preds": preds_chunk}))

            # === 多卡收拢（按序列级）到 rank0 ===
            if use_ddp and world_size > 1:
                gathered = [None for _ in range(world_size)] if rank == 0 else None
                dist.gather_object(local_results, gathered, dst=0)
            else:
                gathered = [local_results]  # 单卡情形

            # === rank0：按时间维拼回整条序列的预测，并评估/落盘 ===
            if rank == 0:
                merged_preds = merge_preds_chunks(gathered)  # dict: 每个键维度0拼成 T
                # print(merged_preds.keys())

                merged_preds_np = tree_to_numpy(merged_preds)
                # === 位姿评估（保持你的原逻辑）===
                cam_cfg = cfg.get("evaluation", {}).get("camera", {})
                if cam_cfg.get("enabled", True):
                    cam_cfg_eff = dict(cam_cfg)
                    if only_rank0_visual:
                        # rank0 可视化开关按照你的配置；其它 rank 本来就不评估
                        pass
                    pose_results_by_mode = evaluate_and_visualize_poses(
                        preds=merged_preds_np,
                        seq_item=seq_item,
                        out_dir=out_dir_this_root,
                        camera_cfg=cam_cfg_eff,
                    )
                    for mode, res in pose_results_by_mode.items():
                        for mname, mdict in res.items():
                            for k, v in mdict.items():
                                if isinstance(v, (int, float)):
                                    pose_collect[mode][mname][k].append(v)

                # === 深度评估（保持你的原逻辑）===
                depth_cfg = cfg.get("evaluation", {}).get("depth", {"enabled": True})
                if depth_cfg.get("enabled", True):
                    depth_cfg_eff = dict(depth_cfg)
                    
                    depth_metrics = evaluate_sequence_depth(
                        preds=merged_preds_np,
                        gt_paths=seq_item.gt.get("depth_paths", []),
                        masks=seq_item.gt.get("valid_masks", None),
                        depth_cfg=depth_cfg_eff,
                        out_dir=out_dir_this_root,
                    )
                    
                    depth_metrics = {
                        k: (float(v) if isinstance(v, (int, float)) or np.isscalar(v) else v)
                        for k, v in depth_metrics.items()
                    }
                    
                    all_seq_depth_metrics.append(depth_metrics)
                else:
                    depth_metrics = {}

                # 每序列的 metrics 快速落盘（保持你的原逻辑）
                dump_json({
                    "sequence": seq_name,
                    "depth": depth_metrics,
                }, out_dir_this_root / "metrics_summary.json")

            # 各 rank 在此序列结束处同步
            if dist_is_ready():
                dist.barrier()

        except Exception as e:
            print(f"[WARN][rank {rank}] sequence {seq_id} failed: {e}")

    if use_ddp and world_size > 1 and dist.is_available() and dist.is_initialized():
        # 先把要传输的对象“净化”为纯可序列化结构
        local_obj = {
            "depth": _to_plain(all_seq_depth_metrics),
            "pose":  _to_plain(pose_collect),
        }
    
        if rank == 0:
            object_gather_list = [None] * world_size
            dist.gather_object(
                obj=local_obj,
                object_gather_list=object_gather_list,
                dst=0,
            )
    
            # 在 rank0 合并
            merged_depth = []
            # 用顶层函数作为 default_factory（可 picklable），避免 lambda
            def dd3(): return defaultdict(list)        # [key] -> list
            def dd2(): return defaultdict(dd3)         # [metric_name] -> dd3
            def dd1(): return defaultdict(dd2)         # [mode] -> dd2
            merged_pose = dd1()
    
            for it in object_gather_list:
                if not it:
                    continue
                
                # depth: 直接拼接
                merged_depth.extend(it.get("depth", []))
    
                # pose: 期望结构 {mode: {metric_name: {k: [vals]}}}
                for mode, groups in it.get("pose", {}).items():
                    for mname, stats in groups.items():
                        for k, arr in stats.items():
                            if arr is None:
                                continue
                            # arr 现在是 list（我们已经 _to_plain 过了）
                            merged_pose[mode][mname][k].extend(arr)
    
            # 覆盖为聚合结果
            all_seq_depth_metrics = merged_depth
            pose_collect = merged_pose
        else:
            # 非 dst 仅发送对象
            dist.gather_object(obj=local_obj, dst=0)
    
        # 若后续还有协作逻辑，做一次同步
        dist.barrier()
    
    # ===== 只有 rank0 写最终汇总（保持你的原逻辑）=====
    if rank == 0:
        summary = {}
        if all_seq_depth_metrics:
            summary["depth_overall_mean"] = aggregate_depth_metrics(all_seq_depth_metrics)
    
        pose_overall = {}
        # pose_collect: {mode: {metric_name: {k: [vals]}}}
        for mode, groups in pose_collect.items():
            pose_overall[mode] = {}
            for mname, stats_dict in groups.items():
                vals = {k: float(sum(v) / max(len(v), 1)) for k, v in stats_dict.items() if len(v) > 0}
                if vals:
                    pose_overall[mode][mname] = vals
        if pose_overall:
            summary["camera_overall_mean"] = pose_overall
    
        dump_json({"summary": summary, "num_sequences": len(loader.list_sequences())}, out_root / "summary.json")
        print("[rank0] Inference done. Summary:", json.dumps(summary, indent=2, ensure_ascii=False))
    
    # ===== DDP 清理（若存在）=====
    if dist_is_ready():
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

def _to_plain(obj):
    """递归把对象转成可通过 gather_object 传输的纯类型（dict/list/num/str）。"""
    # defaultdict -> dict（去掉 default_factory）
    if isinstance(obj, defaultdict):
        obj = dict(obj)

    # dict 递归
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}

    # list / tuple 递归（tuple 也转 list，避免后面有自定义 tuple 子类）
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    # torch.Tensor / np.ndarray -> list
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 其他基础类型直接返回；不可序列化的兜底成 str
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)




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
