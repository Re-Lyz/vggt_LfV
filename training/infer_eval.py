# Copyright (c) Meta Platforms
# Standalone validation inference + evaluation runner (no changes to Trainer).

# ========== 标准库 ==========
import argparse
import json
from collections import defaultdict
from pathlib import Path

# ========== 第三方库 ==========
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import OmegaConf

# ========== 项目内模块 ==========
from eval.depth import (
    aggregate_depth_metrics as agg_depth_metrics,
    evaluate_sequence_depth as eval_depth,
)
from eval.evo_pose import evaluate_and_visualize_poses as eval_poses
from trainer import Trainer
from train_utils.general import copy_data_to_device
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

@torch.no_grad()
def run_val_dump_and_eval_external(
    trainer: Trainer,
    out_dir: str,
    epoch_tag: str = "latest",
    camera_cfg: dict | None = None,
    depth_cfg: dict | None = None,
):
    """
    验证集逐 batch 前向 → 评估（相机+深度）→ 汇总并仅落盘 JSON（不返回）
    - pose：统一为 absT_quaR_FoV 编码（取最后阶段），按 valid_frame_mask 筛帧
    - depth：统一为 [B,T,H,W]，GT resize 到预测分辨率，并生成 depth_mask（finite & >0）
    - 相机评估：APE/RPE，多对齐；深度评估：AbsRel/RMSE/δ
    - 输出：每 batch 的可视化与指标 + 全局 metrics_summary.json
    """
    # 默认评估配置（可外部覆盖）
    if camera_cfg is None:
        camera_cfg = {
            "align": ["sim3", "se3", "none"],
            "quat_order": "xyzw",
            "plot_align_mode": "sim3",
            "plot_3d": True,
            "rpe_delta": 1,
            "rpe_delta_unit": "frames",
        }
    if depth_cfg is None:
        depth_cfg = {"align": "median", "min_depth": 1e-3, "max_depth": 80.0}

    model = trainer.model
    model.eval()
    device = getattr(trainer, "device", "cuda")
    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if use_ddp else 0

    out_root = Path(out_dir) / f"epoch-{epoch_tag}" / f"eval_rank{rank}"
    out_root.mkdir(parents=True, exist_ok=True)

    # DataLoader
    val_ds = getattr(trainer, "val_dataset", None)
    assert val_ds is not None, "val_dataset 未初始化；请确认 data.val 已配置并在 Trainer.__init__ 中被实例化"
    val_loader = val_ds.get_loader(epoch=int(trainer.epoch + trainer.distributed_rank))

    # AMP
    amp_enabled = bool(trainer.optim_conf.amp.enabled)
    amp_type = trainer.optim_conf.amp.amp_dtype
    assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
    amp_dtype = torch.bfloat16 if amp_type == "bfloat16" else torch.float16

    # 汇总容器
    per_batch_camera: list[dict] = []
    per_batch_depth:  list[dict] = []

    for data_iter, batch in enumerate(val_loader):
        try:
            # 预处理
            with torch.cuda.amp.autocast(enabled=False):
                batch = trainer._process_batch(batch)
            batch = copy_data_to_device(batch, device, non_blocking=True)

            # 前向
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                preds = model(images=batch["images"])

            # ========= Pose（编码 & 筛帧）=========
            if "pose_enc_list" not in preds:
                raise KeyError("preds 缺少 'pose_enc_list'（相机评估需要该键）")
            pred_pose_enc = preds["pose_enc_list"][-1]  # [B,T,D]

            imgs = batch["images"]
            if imgs.ndim == 5:     # [B,T,C,H,W]
                B, T, _, Hp_img, Wp_img = imgs.shape
            elif imgs.ndim == 4:   # [B,C,H,W] 视作 T=1
                B, _, Hp_img, Wp_img = imgs.shape
                T = pred_pose_enc.shape[1] if pred_pose_enc.ndim >= 3 else 1
            else:
                raise RuntimeError(f"不支持的 images 形状：{imgs.shape}")

            gt_pose_enc = extri_intri_to_pose_encoding(
                extrinsics=batch["extrinsics"],
                intrinsics=batch["intrinsics"],
                image_size_hw=(Hp_img, Wp_img),
                pose_encoding_type="absT_quaR_FoV",
            )

            point_masks = batch.get("point_masks", None)
            if point_masks is not None:
                valid_frame_mask = (point_masks[:, 0].sum(dim=[-1, -2]) > 100)  # [B,T]
                pred_pose_enc_valid = pred_pose_enc[valid_frame_mask]
                gt_pose_enc_valid   = gt_pose_enc[valid_frame_mask]
            else:
                pred_pose_enc_valid = pred_pose_enc
                gt_pose_enc_valid   = gt_pose_enc

            # ========= Depth（对齐分辨率 & 掩码）=========
            if "depth" in preds:
                pred_depth = preds["depth"]
            elif "depth_list" in preds:
                pred_depth = preds["depth_list"][-1]
            else:
                raise KeyError("preds 缺少 'depth' / 'depth_list'")

            # -> [B,T,H,W]
            if pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth.squeeze(-1)
            elif pred_depth.ndim == 4 and imgs.ndim == 4:
                if pred_depth.shape[1] == 1:
                    pred_depth = pred_depth.squeeze(1)
                pred_depth = pred_depth.unsqueeze(1)
            elif pred_depth.ndim != 4:
                raise RuntimeError(f"不支持的 depth 形状：{pred_depth.shape}")

            gt_depth = batch.get("depths", None)
            if gt_depth is not None:
                if gt_depth.ndim == 5 and gt_depth.shape[-1] == 1:
                    gt_depth = gt_depth.squeeze(-1)
                elif gt_depth.ndim == 4 and imgs.ndim == 4:
                    if gt_depth.shape[1] == 1:
                        gt_depth = gt_depth.squeeze(1)
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim != 4:
                    raise RuntimeError(f"不支持的 depths 形状：{gt_depth.shape}")

            Bh, Th, Hp, Wp = pred_depth.shape
            if gt_depth is not None and gt_depth.shape[-2:] != (Hp, Wp):
                gt_depth = F.interpolate(
                    gt_depth.flatten(0, 1).unsqueeze(1), size=(Hp, Wp), mode="nearest"
                ).squeeze(1).view(Bh, Th, Hp, Wp)

            depth_mask = None
            if gt_depth is not None:
                depth_mask = torch.isfinite(gt_depth) & (gt_depth > 0)

            # ========= 评估 =========
            batch_out_dir = out_root / f"batch_{data_iter:06d}"
            (batch_out_dir / "camera").mkdir(parents=True, exist_ok=True)
            (batch_out_dir / "depth").mkdir(parents=True, exist_ok=True)

            cam_metrics = eval_poses(
                pred_pose_enc_valid=pred_pose_enc_valid,
                gt_pose_enc_valid=gt_pose_enc_valid,
                out_dir=batch_out_dir / "camera",
                camera_cfg=camera_cfg,
            )
            per_batch_camera.append(cam_metrics)

            if gt_depth is not None:
                depth_metrics = eval_depth(
                    pred_depth=pred_depth,     # [B,T,H,W] 或 [S,H,W] 或 List[H,W]
                    gt_depth=gt_depth,         # 同上；若没有 GT 就不要调用评估
                    depth_mask=depth_mask,     # 可选；>0 为有效像素
                    depth_cfg=depth_cfg,
                    out_dir=batch_out_dir / "depth",
                )
                per_batch_depth.append(depth_metrics)

        except Exception as e:
            print(f"[WARN][rank {rank}] batch {data_iter} failed: {e}")
            continue

    # ========= 汇总 Summary（跨 batch）=========
    summary = {}

    def _merge_camera_results(all_res: list[dict]) -> dict:
        if not all_res:
            return {}
        modes = {}
        for res in all_res:
            for mode, block in res.items():
                m = modes.setdefault(mode, {"APE": [], "RPE": []})
                if "APE" in block and block["APE"]:
                    m["APE"].append(block["APE"])
                if "RPE" in block and block["RPE"]:
                    m["RPE"].append(block["RPE"])
        merged = {}
        for mode, parts in modes.items():
            merged[mode] = {}
            for key in ("APE", "RPE"):
                if not parts[key]:
                    merged[mode][key] = {}
                    continue
                keys = set().union(*[d.keys() for d in parts[key]])
                agg = {}
                for k in keys:
                    vals = []
                    for d in parts[key]:
                        v = d.get(k, np.nan)
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(np.nan)
                    agg[k] = float(np.nanmean(vals))
                merged[mode][key] = agg
        return merged

    cam_summary = _merge_camera_results(per_batch_camera)
    if cam_summary:
        summary["camera"] = cam_summary

    if per_batch_depth:
        depth_summary = agg_depth_metrics(per_batch_depth)
        summary["depth"] = depth_summary

    # 写 JSON（不返回）
    summary_path = out_root / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 控制台简报
    def _fmt(v):
        try: return f"{float(v):.4f}"
        except Exception: return str(v)

    print("\n===== Validation Summary =====")
    if "camera" in summary:
        for mode in (m for m in ("sim3", "se3", "none", "scale") if m in summary["camera"]):
            ape = summary["camera"][mode].get("APE", {})
            rpe = summary["camera"][mode].get("RPE", {})
            print(f"[Camera][{mode}]  APE_trans_rmse={_fmt(ape.get('trans_rmse'))}  "
                  f"APE_rot_rmse_deg={_fmt(ape.get('rot_rmse_deg'))}  "
                  f"RPE_trans_rmse={_fmt(rpe.get('trans_rmse'))}  "
                  f"RPE_rot_rmse_deg={_fmt(rpe.get('rot_rmse_deg'))}")
            break
    if "depth" in summary:
        d = summary["depth"]
        print(f"[Depth]  AbsRel={_fmt(d.get('abs_rel'))}  RMSE={_fmt(d.get('rmse'))}  "
              f"δ1={_fmt(d.get('d1'))}  δ2={_fmt(d.get('d2'))}  δ3={_fmt(d.get('d3'))}")

    # 清理
    del val_loader
    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser(description="Validate + Evaluate with Trainer & Eval configs")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="TRAIN config name (without .yaml), e.g., default",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="default",
        help="EVAL config name under config/eval (without .yaml), e.g., default",
    )
    args = parser.parse_args()

    # 组合两个配置：主配置 + 评估配置（作为一个名为 eval 的组）
    # 你项目的 config 根目录是 'config'；'eval/<name>.yaml' 会被合成到 cfg.eval
    with initialize(version_base=None, config_path="config"):
        cfg = compose(
            config_name=args.config,
            overrides=[f"eval={args.eval_config}"],  # 也可省略，走默认 default
        )

    # 取出评估配置块（dict）
    # 注意：OmegaConf -> python dict，便于传给评估函数
    if "eval" not in cfg:
        raise KeyError("未找到 cfg.eval；请确认存在 config/eval/<name>.yaml 且传入 --eval_config。")

    eval_cfg = cfg.eval
    camera_cfg = OmegaConf.to_container(eval_cfg.get("camera", {}), resolve=True)
    depth_cfg  = OmegaConf.to_container(eval_cfg.get("depth",  {}), resolve=True)

    # out_dir 支持引用主 cfg 的字段（例如 ${exp_name}），Hydra 会在 compose 时解析
    out_dir   = eval_cfg.get("out_dir", f"outputs/eval/{cfg.get('exp_name', 'exp')}")
    epoch_tag = eval_cfg.get("epoch_tag", "latest")

    # 1) 实例化 Trainer（构建模型/数据）
    trainer = Trainer(**cfg, mode="test")

    # 2) 直接做验证前向 + 评估 + 落盘（函数内部不 return，只写 JSON）
    run_val_dump_and_eval_external(
        trainer,
        out_dir=out_dir,
        epoch_tag=epoch_tag,
        camera_cfg=camera_cfg,
        depth_cfg=depth_cfg,
    )


if __name__ == "__main__":
    main()
