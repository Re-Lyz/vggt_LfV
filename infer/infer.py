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

# -----------------------
# 占位函数导入（尚未实现）
# -----------------------
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


# -----------------------
# 公共：一次前向（推理）
# -----------------------
# === 修改：单步前向，兼容 DataParallel ===
def _forward_infer(
    model,
    images: torch.Tensor,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    query_points: Optional[torch.Tensor] = None,
):
    with torch.no_grad():
        if device.startswith("cuda") and amp_enabled:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                # DataParallel: 直接调用 model(images, query_points)
                if isinstance(model, nn.DataParallel):
                    preds = model(images, query_points)
                else:
                    preds = model.infer_images(images, query_points=query_points)
        else:
            if isinstance(model, nn.DataParallel):
                preds = model(images, query_points)
            else:
                preds = model.infer_images(images, query_points=query_points)
    return preds


# -----------------------
# 推理主流程
# -----------------------
def run_inference(cfg, device: str, amp_dtype: torch.dtype, amp_enabled: bool, args=None):
    out_root = Path(cfg.get("output_dir", "outputs/infer"))
    ensure_dir(out_root)
    dump_config(cfg, out_root)

    # 读取是否启用多卡
    dp_cfg = cfg.get("inference", {}).get("data_parallel", True)  # 默认开
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # 模型：先构建单卡 wrapper
    # 注意：为了 DP 正确 scatter/gather，主设备固定到 cuda:0
    main_device = "cuda:0" if (device.startswith("cuda") and n_gpu > 0) else device
    model = VGGTWrapper(device=main_device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)

    # 如果具备多卡且允许 DP，则包一层适配器 + DataParallel
    if dp_cfg and n_gpu > 1 and main_device.startswith("cuda"):
        model = nn.DataParallel(_DPInferAdapter(model), device_ids=list(range(n_gpu)))
        # 之后所有张量一律先放到 cuda:0，再交给 DP 切分
        device = main_device

    # 数据集（要求 loader 自带 load_and_preprocess_images）
    loader = get_dataset_loader(cfg)

    # print(loader.base.get_data())

    # 调试模式（需判空 args）
    if args and getattr(args, "debug_loader", False):
        out_dir = Path(cfg.get("output_dir", "outputs")) / "debug_loader"
        ensure_dir(out_dir)
        inspect_loader(
            loader=loader,
            device=device,
            out_dir=out_dir,
            max_sequences=getattr(args, "debug_max_seqs", 1),
            max_frames_visual=getattr(args, "debug_max_frames", 12),
            fps=int(cfg.get("visualization", {}).get("video_overlay", {}).get("fps", 8)),
        )
        return  # 调试完直接退出

    seq_ids = loader.list_sequences()
    limit = cfg.get("dataset", {}).get("limit_seqs", -1)
    if isinstance(limit, int) and limit > 0:
        seq_ids = seq_ids[:limit]

    all_seq_depth_metrics = []
    pose_collect = defaultdict(lambda: {"APE": defaultdict(list), "RPE": defaultdict(list)})

    pbar = tqdm(seq_ids, desc=f"Inference ({cfg.get('dataset', {}).get('name', '')})")
    for seq_id in pbar:
        try:
            seq_item = loader.build_sequence(seq_id)  # 需含 images / gt / meta
            seq_out = out_root / seq_item.meta.get("sequence_name", "seq")
            ensure_dir(seq_out)

            # 预处理由数据集完成
            images = loader.load_and_preprocess_images(seq_item.images, device=device)  # 若启用DP，此处 device 为 "cuda:0"
            query_points = None
            if hasattr(loader, "load_and_preprocess_query_points"):
                try:
                    query_points = loader.load_and_preprocess_query_points(seq_item, device=device)
                except Exception:
                    query_points = None
            elif isinstance(getattr(seq_item, "gt", None), dict):
                qp = seq_item.gt.get("query_points_tensor", None)
                query_points = qp if isinstance(qp, torch.Tensor) else None

            # 推理
            preds = _forward_infer(
                model, images,
                device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                query_points=query_points,
            )
            print("finished infer", seq_id, {k: (v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in preds.items()})
            # print()
            # print("pose:", preds.get("pose", None).shape)
            # print("depth:", preds.get("depth", None).shape)
            # print("depth_conf:", preds.get("depth_conf", None).shape)
            # print()
            
            # —— 位姿评估（按多种对齐模式）——
            cam_cfg = cfg.get("evaluation", {}).get("camera", {})
            pose_results_by_mode = {}
            if cam_cfg.get("enabled", True):
                pose_results_by_mode = evaluate_and_visualize_poses(
                    preds=preds,
                    seq_item=seq_item,
                    out_dir=seq_out,
                    camera_cfg=cam_cfg,   # 里边读取 align / rpe_delta 等
                )
                # 收集到全局 pose_collect，用于最后做 overall mean
                for mode, res in pose_results_by_mode.items():            # mode: 'sim3'/'se3'/'none'/...
                    for mname, mdict in res.items():                      # 'APE' 或 'RPE'
                        for k, v in mdict.items():
                            if isinstance(v, (int, float)):               # 只聚合数值项
                                pose_collect[mode][mname][k].append(v)

            # —— 深度评估 —— 
            depth_cfg = cfg.get("evaluation", {}).get("depth", {"enabled": True})
            depth_metrics = {}
            if depth_cfg.get("enabled", True):
                depth_metrics = evaluate_sequence_depth(
                    preds=preds,
                    gt_paths=seq_item.gt.get("depth_paths", []),
                    masks=seq_item.gt.get("valid_masks", None),
                    depth_cfg=depth_cfg,
                    out_dir=seq_out,
                )
                all_seq_depth_metrics.append(depth_metrics)
                # print("depth_metrics:", depth_metrics)
            # 每序列的快速汇总（可选）
            dump_json({
                "sequence": seq_item.meta.get("sequence_name", str(seq_id)),
                "pose": pose_results_by_mode,
                "depth": depth_metrics,
            }, seq_out / "metrics_summary.json")

        except Exception as e:
            # 单序列出错不中断，标记并继续
            print(f"[WARN] sequence {seq_id} failed: {e}")
            
    # 可视化实现


    # ===== 汇总 =====
    summary = {}

    # 深度：整体均值
    if all_seq_depth_metrics:
        summary["depth_overall_mean"] = aggregate_depth_metrics(all_seq_depth_metrics)

    # 位姿：按模式/指标聚合均值
    pose_overall = {}
    for mode, groups in pose_collect.items():
        pose_overall[mode] = {}
        for mname, stats_dict in groups.items():  # 'APE' 或 'RPE'
            pose_overall[mode][mname] = {}
            for k, arr in stats_dict.items():
                if len(arr) == 0:
                    continue
                pose_overall[mode][mname][k] = float(sum(arr) / len(arr))
    if pose_overall:
        summary["camera_overall_mean"] = pose_overall

    dump_json({"summary": summary, "num_sequences": len(seq_ids)}, out_root / "summary.json")
    print("Inference done. Summary:", json.dumps(summary, indent=2, ensure_ascii=False))

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
