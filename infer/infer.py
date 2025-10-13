#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference & Finetune backbone (skeleton only).

- 读取 YAML 配置
- 设定随机种子 / 设备与 AMP 策略
- 构建模型（占位）
- 构建数据加载器（占位）
- mode=inference: 由数据集完成 load_and_preprocess_images -> 推理(占位) -> 评估(占位) -> 可视化(占位)
- mode=finetune : 训练主循环占位（优化器/调度器/训练步/评估步均为占位）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from tqdm import tqdm
import numpy as np
import torch

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
    # build_optimizer,       # (model:VGGTWrapper, cfg:dict) -> torch.optim.Optimizer     (占位)
    # build_scheduler,       # (optimizer, cfg:dict) -> torch.optim.lr_scheduler._LRScheduler or None  (占位)
    # training_step,         # (model, batch, device, scaler_or_none, amp_dtype, cfg) -> Dict[str,float]  (占位)
    # eval_step,             # (model, batch, device, cfg) -> Dict[str,float]             (占位)
    # save_checkpoint,       # (model, optimizer, scheduler, epoch, out_dir:Path) -> None  (占位)
)

# data：数据加载器工厂（推理 & 训练的占位接口）
from data.dataloader import (
    get_dataset_loader,      # (cfg:dict) -> obj: 需实现 .list_sequences() / .build_sequence(seq_id) / .load_and_preprocess_images(paths, device)
    get_train_dataloader,    # (cfg:dict) -> torch.utils.data.DataLoader                 (占位)
    get_eval_dataloader,     # (cfg:dict) -> torch.utils.data.DataLoader                 (占位)
)

from tools.debug_loader import inspect_loader  # (loader, device, out_dir, max_sequences, max_frames_visual, fps) -> None

# eval：评估指标与聚合（推理用的占位）
# from eval.metrics import (
#     evaluate_sequence_depth,   # (pred_depths:List[np.ndarray], gt_paths:List[str], masks:Optional[List[str]], depth_cfg:dict) -> Dict[str,float]
#     aggregate_depth_metrics,   # (per_seq_metrics:List[Dict[str,float]]) -> Dict[str,float]
# )

# vis：可视化（占位）
# from vis.camera_path import (
#     extract_cam_centers,       # (predictions:Dict[str,Any]) -> Optional[np.ndarray[B,3]]
#     save_3d_path_plot,         # (centers:np.ndarray[B,3], out_path:Path) -> None
# )
# from vis.video_overlay import (
#     make_video_with_overlays,  # (image_paths:List[str], out_path:Path, traj_normed_xy:Optional[np.ndarray[B,2]], fps:int, draw_traj:bool, draw_cam_axis:bool, axis_len:int) -> None
# )

# -----------------------
# 公共：一次前向（推理）
# -----------------------
def _forward_infer(
    model: VGGTWrapper,
    images: torch.Tensor,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    query_points: Optional[torch.Tensor] = None,    
):
    with torch.no_grad():
        if device == "cuda" and amp_enabled:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                preds = model.infer_images(images, query_points=query_points)  
        else:
            preds = model.infer_images(images, query_points=query_points) 
    return preds

# -----------------------
# 推理主流程
# -----------------------
def run_inference(cfg: Dict[str, Any], device: str, amp_dtype: torch.dtype, amp_enabled: bool, args=None):
    out_root = Path(cfg.get("output_dir", "outputs/infer"))
    ensure_dir(out_root)
    dump_config(cfg, out_root)

    # 模型
    model = VGGTWrapper(device=device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)

    # 数据集（要求 loader 自带 load_and_preprocess_images）
    loader = get_dataset_loader(cfg)
    
    if args.debug_loader:
        out_dir = Path(cfg.get("output_dir", "outputs")) / "debug_loader"
        ensure_dir(out_dir)
        inspect_loader(
            loader=loader,
            device=device,
            out_dir=out_dir,
            max_sequences=args.debug_max_seqs,
            max_frames_visual=args.debug_max_frames,
            fps=int(cfg.get("visualization",{}).get("video_overlay",{}).get("fps", 8)),
        )
        return  # 调试完直接退出

    seq_ids: List[str] = loader.list_sequences()
    limit = cfg.get("dataset", {}).get("limit_seqs", -1)
    if isinstance(limit, int) and limit > 0:
        seq_ids = seq_ids[:limit]

    all_seq_metrics: List[Dict[str, float]] = []
    depth_cfg = cfg.get("evaluation", {}).get("depth", {"enabled": True})

    pbar = tqdm(seq_ids, desc=f"Inference ({cfg.get('dataset', {}).get('name', '')})")
    for seq_id in pbar:
        seq_item = loader.build_sequence(seq_id)  # 需含 images / gt / meta
        seq_out = out_root / seq_item.meta.get("sequence_name", "seq")
        ensure_dir(seq_out)

        # 预处理由数据集完成
        images = loader.load_and_preprocess_images(seq_item.images, device=device)
        
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
        
        print("finished infer", seq_id, {k: (v.shape if isinstance(v, torch.Tensor) else type(v)) for k,v in preds.items()})
    #     # 评估（以深度为例）
    #     if depth_cfg.get("enabled", True) and seq_item.gt.get("depth_paths"):
    #         pred_depths = preds.get("depths", None)
    #         if isinstance(pred_depths, torch.Tensor):
    #             if pred_depths.ndim == 4:   # [B,1,H,W]
    #                 pred_depths = [pred_depths[i, 0].float().cpu().numpy() for i in range(pred_depths.shape[0])]
    #             elif pred_depths.ndim == 3: # [B,H,W]
    #                 pred_depths = [pred_depths[i].float().cpu().numpy() for i in range(pred_depths.shape[0])]
    #         elif isinstance(pred_depths, (list, tuple)):
    #             pred_depths = [d.detach().float().cpu().numpy() if hasattr(d,"detach") else np.asarray(d)
    #                            for d in pred_depths]

    #         if pred_depths is not None:
    #             m = evaluate_sequence_depth(
    #                 pred_depths=pred_depths,
    #                 gt_paths=seq_item.gt.get("depth_paths", []),
    #                 masks=seq_item.gt.get("valid_masks", None),
    #                 depth_cfg=depth_cfg,
    #             )
    #             all_seq_metrics.append(m)
    #             dump_json({"sequence": seq_item.meta, "metrics": m}, seq_out / "metrics_depth.json")
    #             pbar.set_postfix(abs_rel=f"{m.get('abs_rel', float('nan')):.3f}",
    #                              rmse=f"{m.get('rmse', float('nan')):.3f}")

    #     # 可视化
    #     cam_centers = extract_cam_centers(preds)  # Optional[np.ndarray[B,3]]
    #     if cam_centers is not None and cam_centers.shape[0] >= 2:
    #         save_3d_path_plot(cam_centers, seq_out / "camera_path_3d.png")
    #         xy = cam_centers[:, :2]
    #         xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-6)
    #     else:
    #         xy = None

    #     vis_cfg = cfg.get("visualization", {}).get("video_overlay", {})
    #     if vis_cfg.get("enabled", True):
    #         make_video_with_overlays(
    #             image_paths=seq_item.images,
    #             out_path=seq_out / "overlay.mp4",
    #             traj_normed_xy=xy,
    #             fps=int(vis_cfg.get("fps", 10)),
    #             draw_traj=bool(vis_cfg.get("draw_traj", True)),
    #             draw_cam_axis=bool(vis_cfg.get("draw_cam_axis", True)),
    #             axis_len=int(vis_cfg.get("axis_len", 40)),
    #         )

    # # 汇总
    # summary = {}
    # if all_seq_metrics:
    #     summary["depth_overall_mean"] = aggregate_depth_metrics(all_seq_metrics)
    # dump_json({"summary": summary, "num_sequences": len(seq_ids)}, out_root / "summary.json")
    # print("Inference done. Summary:", json.dumps(summary, indent=2, ensure_ascii=False))

# -----------------------
# 微调主流程（占位）
# -----------------------
# def run_finetune(cfg: Dict[str, Any], device: str, amp_dtype: torch.dtype, amp_enabled: bool):
#     """
#     训练骨架，仅保留接口调用。实际实现请在:
#       - models.vggt_wrapper.{build_optimizer, build_scheduler, training_step, eval_step, save_checkpoint}
#       - data.dataloader.{get_train_dataloader, get_eval_dataloader}
#     中补齐。
#     """
#     out_root = Path(cfg.get("output_dir", "outputs/finetune"))
#     ensure_dir(out_root)
#     dump_config(cfg, out_root)

#     # 模型
#     model = VGGTWrapper(device=device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)

#     # DataLoader（占位）
#     train_loader = get_train_dataloader(cfg)  # 需你实现
#     eval_loader  = get_eval_dataloader(cfg)   # 需你实现

#     # 优化器/调度器（占位）
#     optimizer = build_optimizer(model, cfg)   # 需你实现
#     scheduler = build_scheduler(optimizer, cfg)  # 可选

#     scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and amp_enabled))

#     train_cfg = cfg.get("train", {})
#     epochs = int(train_cfg.get("epochs", 1))
#     log_interval = int(train_cfg.get("log_interval", 10))
#     eval_interval = int(train_cfg.get("eval_interval", 1))
#     ckpt_interval = int(train_cfg.get("ckpt_interval", 1))

#     global_step = 0
#     for epoch in range(1, epochs + 1):
#         model.model.train()
#         pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch}/{epochs}")
#         for step, batch in enumerate(pbar, start=1):
#             # 单步训练（占位）
#             stats = training_step(model, batch, device, scaler, amp_dtype, cfg)  # 返回 {'loss':..., ...}
#             global_step += 1

#             if step % log_interval == 0 and isinstance(stats, dict):
#                 pbar.set_postfix({k: f"{v:.4f}" for k, v in stats.items() if isinstance(v, (int, float))})

#         # 学习率调度（占位）
#         if scheduler is not None:
#             try:
#                 scheduler.step()
#             except Exception:
#                 pass

#         # 评估（占位）
#         if eval_interval > 0 and (epoch % eval_interval == 0):
#             model.model.eval()
#             eval_stats_accum: Dict[str, float] = {}
#             eval_count = 0
#             with torch.no_grad():
#                 for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
#                     estats = eval_step(model, batch, device, cfg)  # 返回 dict
#                     if isinstance(estats, dict):
#                         for k, v in estats.items():
#                             eval_stats_accum[k] = eval_stats_accum.get(k, 0.0) + float(v)
#                         eval_count += 1
#             if eval_count > 0:
#                 eval_stats_mean = {k: v / eval_count for k, v in eval_stats_accum.items()}
#                 dump_json({"epoch": epoch, "eval": eval_stats_mean, "step": global_step}, out_root / f"eval_epoch_{epoch:03d}.json")
#                 print(f"[Eval][Epoch {epoch}] {json.dumps(eval_stats_mean, indent=2, ensure_ascii=False)}")

#         # 保存检查点（占位）
#         if ckpt_interval > 0 and (epoch % ckpt_interval == 0):
#             save_checkpoint(model, optimizer, scheduler, epoch, out_root)

#     print("Finetune (skeleton) finished.")

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
    elif mode == "finetune":
        # run_finetune(cfg, device, amp_dtype, amp_enabled)
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")

if __name__ == "__main__":
    main()
