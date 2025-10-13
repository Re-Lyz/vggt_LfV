# eval/evo_pose.py（替换 evaluate_and_visualize_poses）
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import json, os
import torch
import matplotlib.pyplot as plt
from evo.core.metrics import APE, RPE, PoseRelation, Unit
from mpl_toolkits.mplot3d import Axes3D  
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from evo.core.metrics import APE, RPE, PoseRelation
    from evo.tools import plot as evo_plot
    EVO_AVAILABLE = True
    _EVO_IMPORT_ERROR = None
except Exception as e:
    EVO_AVAILABLE = False
    _EVO_IMPORT_ERROR = e
    

def _poses_from_outputs(preds: Dict[str, Any]) -> np.ndarray:
    """
    只接受 preds["pose"]，形状 [B=1, S, 9] 或 [S, 9]。
    9D = rot6d(6) + t(3)，直接解释为 c2w。
    返回：c2w，形状 [S, 4, 4]
    """
    # print("preds keys:", preds.keys())
    
    if "pose_enc" not in preds:
        raise KeyError("[evo_pose] 需要 preds['pose_enc']（9D），但未找到该键。")

    P = _to_np(preds["pose_enc"])
    if P.ndim == 3 and P.shape[0] == 1:
        P = P[0]  # [S, 9]
    if P.ndim != 2 or P.shape[1] != 9:
        raise ValueError(f"[evo_pose] preds['pose'] 形状应为 [1,S,9] 或 [S,9]，当前为 {P.shape}。")

    rots6d = P[:, :6]                    # [S, 6]
    trans  = P[:, 6:9]                   # [S, 3]
    R = _rot6d_to_R(rots6d)              # [S, 3, 3]

    S = P.shape[0]
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None, ...], S, axis=0)  # [S, 4, 4]
    c2w[:, :3, :3] = R
    c2w[:, :3,  3] = trans
    return c2w

def _poses_from_pose_txt(pose_txt: str, fmt: str = "w2c") -> Optional[np.ndarray]:
    """从 pose.txt 读取为 [S, 4, 4]（支持逗号或空白分隔）。fmt ∈ {'w2c','c2w'}。"""
    import re
    from pathlib import Path
    import numpy as np

    p = Path(pose_txt)
    if not p.is_file():
        return None

    mats = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # 同时支持逗号/空白分隔
            toks = re.split(r"[,\s]+", s)
            toks = [t for t in toks if t != ""]
            try:
                vals = [float(t) for t in toks]
            except ValueError as e:
                raise ValueError(
                    f"[evo_pose] 解析 {pose_txt} 行失败：{s!r}；请确认分隔符或数值格式。"
                ) from e

            if len(vals) == 12:
                M = np.array(vals, dtype=np.float64).reshape(3, 4)
                M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])
            elif len(vals) == 16:
                M = np.array(vals, dtype=np.float64).reshape(4, 4)
            else:
                # 既不是 3x4 也不是 4x4，跳过该行
                continue
            col_t = M[:3, 3]
            row_t = M[3, :3]
            is_row_t = (np.allclose(col_t, 0.0, atol=1e-8) and not np.allclose(row_t, 0.0, atol=1e-8))

            if is_row_t:
                # 列主 -> 行主
                R_colmajor = M[:3, :3]          # 列主的 R
                R_rowmajor = R_colmajor.T       # 转成行主
                t_row = row_t.copy()            # [tx, ty, tz] 存在行里
                M_fixed = np.eye(4, dtype=np.float64)
                M_fixed[:3, :3] = R_rowmajor
                M_fixed[:3,  3] = t_row
                M = M_fixed
                
            mats.append(M)

    if not mats:
        return None

    T = np.stack(mats, axis=0)  # [S, 4, 4]

    fmt_l = (fmt or "w2c").lower()
    if fmt_l == "c2w":
        return T
    elif fmt_l == "w2c":
        # w2c -> c2w（刚体矩阵的显式求逆）
        R = T[:, :3, :3]
        t = T[:, :3, 3:4]
        Rt = np.transpose(R, (0, 2, 1))
        c2w = np.zeros_like(T)
        c2w[:, :3, :3] = Rt
        c2w[:, :3, 3:4] = -Rt @ t
        c2w[:, 3, 3] = 1.0
        return c2w
    else:
        # 未知 fmt，按原样返回
        return T

def _to_np(x):

    try:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _rot6d_to_R(rots6d_np):
    """
    rots6d_np: [..., 6]  ->  [..., 3, 3]
    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    """
    a1 = rots6d_np[..., 0:3]
    a2 = rots6d_np[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2_proj = (b1 * np.sum(b1 * a2, axis=-1, keepdims=True))
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)  # [..., 3, 3]
    return R

def _c2w_to_evo_traj(c2w: np.ndarray):
    """把 [S,4,4] c2w 转为 evo PoseTrajectory3D（四元数为 w,x,y,z）。"""
    import numpy as np
    from evo.core.trajectory import PoseTrajectory3D

    S = c2w.shape[0]
    positions_xyz = c2w[:, :3, 3]          # [S,3]
    R = c2w[:, :3, :3]                      # [S,3,3]

    # 将旋转矩阵批量转四元数（优先 scipy，退化到 evo.transformations）
    def _rots_to_quat_wxyz(Rs: np.ndarray) -> np.ndarray:
        try:
            from scipy.spatial.transform import Rotation as R_
            q_xyzw = R_.from_matrix(Rs).as_quat()            # [S,4] (x,y,z,w)
            q_wxyz = np.concatenate([q_xyzw[:, -1:], q_xyzw[:, :3]], axis=1)
            return q_wxyz.astype(np.float64, copy=False)
        except Exception:
            from evo.core import transformations as tf
            out = []
            for i in range(Rs.shape[0]):
                M = np.eye(4, dtype=np.float64)
                M[:3, :3] = Rs[i]
                q_wxyz = tf.quaternion_from_matrix(M)        # (w,x,y,z)
                out.append(q_wxyz)
            return np.asarray(out, dtype=np.float64)

    quats_wxyz = _rots_to_quat_wxyz(R)

    # 时间戳：若无真实时间，按帧序号填充
    stamps = np.arange(S, dtype=float)

    return PoseTrajectory3D(
        positions_xyz=positions_xyz,
        orientations_quat_wxyz=quats_wxyz,
        timestamps=stamps,
    )

def _parse_align_modes(camera_cfg: Dict[str, Any]) -> List[str]:
    a = camera_cfg.get("align", "sim3")
    if isinstance(a, str):
        return [a.lower()]
    if isinstance(a, (list, tuple)):
        return [str(x).lower() for x in a]
    return ["sim3"]

def _apply_align_settings(metric, mode: str):
    """
    按模式设置 evo 的对齐/尺度校正开关。
    兼容不同版本：优先 metric.settings.align / .correct_scale；
    否则尝试 metric.align / metric.correct_scale（有些版本是直接属性）。
    """
    mode = (mode or "").lower()
    if mode == "none":
        align, correct_scale = False, False
    elif mode == "se3":
        align, correct_scale = True, False
    elif mode == "sim3":
        align, correct_scale = True, True
    elif mode == "scale":
        align, correct_scale = False, True
    else:
        align, correct_scale = True, True  # fallback

    # v1: metric.settings.align / correct_scale
    if hasattr(metric, "settings") and metric.settings is not None:
        try:
            metric.settings.align = align
            metric.settings.correct_scale = correct_scale
            return
        except Exception:
            pass

    # v2: metric.align / correct_scale
    if hasattr(metric, "align"):
        try:
            metric.align = align
        except Exception:
            pass
    if hasattr(metric, "correct_scale"):
        try:
            metric.correct_scale = correct_scale
        except Exception:
            pass

def _umeyama_sim3(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """
    求 s,R,t 使得  s * R @ Y + t ≈ X
    X, Y: [N,3]
    返回: s(float), R(3,3), t(3,)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.shape == Y.shape and X.shape[1] == 3 and X.shape[0] >= 3

    n = X.shape[0]
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    cov = (Yc.T @ Xc) / n                      # 3x3
    U, S, Vt = np.linalg.svd(cov)
    C = np.ones(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        C[-1] = -1.0
    R = U @ np.diag(C) @ Vt

    if with_scale:
        varY = (Yc * Yc).sum() / n
        s = (S * C).sum() / max(varY, 1e-12)   # tr(D*C)/varY
    else:
        s = 1.0

    t = muX - s * (R @ muY)
    return float(s), R, t

def _align_pred_for_plot(gt_xyz: np.ndarray, pred_xyz: np.ndarray, mode: str = "sim3"):
    """
    仅用于绘图对齐：返回对齐后的 pred_xyz_vis。
    mode: 'sim3' | 'se3' | 'scale' | 'none'
    """
    mode = (mode or "sim3").lower()
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)

    if pred_xyz.shape[0] < 3 or gt_xyz.shape[0] < 3:
        return pred_xyz  # 点太少就不对齐

    try:
        if mode == "sim3":
            s, R, t = _umeyama_sim3(gt_xyz, pred_xyz, with_scale=True)
            return (s * (pred_xyz @ R.T) + t)
        elif mode == "se3":
            s, R, t = _umeyama_sim3(gt_xyz, pred_xyz, with_scale=False)
            return (pred_xyz @ R.T) + t
        elif mode == "scale":
            # 仅尺度：用路径长度比做一个稳健近似
            eps = 1e-12
            lg = np.linalg.norm(gt_xyz[-1] - gt_xyz[0]) + eps
            lp = np.linalg.norm(pred_xyz[-1] - pred_xyz[0]) + eps
            s = lg / lp
            return (pred_xyz - pred_xyz.mean(0)) * s + gt_xyz.mean(0)
        else:
            return pred_xyz
    except Exception:
        return pred_xyz
        
def evaluate_and_visualize_poses(
    preds: Dict[str, Any],
    seq_item,
    out_dir: Path,
    camera_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    用 evo 评估位姿，支持多种对齐模式:
      align: "none" | "se3" | "sim3" | "scale" 或列表
    返回: { "<mode>": { "APE": {...}, "RPE": {...} }, ... }
    同时保存：轨迹XY/3D图；误差曲线若可提取则绘制（提取不到则跳过且不报错）。
    """
    if not EVO_AVAILABLE:
        print(f"[evo_pose] 未安装 evo，跳过相机评估：{_EVO_IMPORT_ERROR}")
        return {}


    # -------- 读取预测与GT --------
    pred_c2w = _poses_from_outputs(preds)
    if pred_c2w is None:
        print("[evo_pose] preds 中未找到位姿（c2w/w2c），跳过。")
        return {}

    gt_txt = seq_item.meta.get("pose_txt", None)
    if not gt_txt:
        print("[evo_pose] 未提供 GT pose_txt，跳过。")
        return {}

    gt_fmt     = str(camera_cfg.get("gt_pose_format", "w2c")).lower()
    plot_3d    = bool(camera_cfg.get("plot_3d", True))
    rpe_delta  = int(camera_cfg.get("rpe_delta", 1))
    rpe_unit_u = str(camera_cfg.get("rpe_delta_unit", "frames")).lower()

    # 统一到 Unit 枚举
    def _normalize_rpe_unit_enum(u: str):
        if u in ("frames", "frame", "f", "index", "indices", "idx", "samples"):
            return Unit.frames
        if u in ("seconds", "second", "s", "time", "t"):
            return Unit.seconds
        return Unit.frames
    rpe_unit_enum = _normalize_rpe_unit_enum(rpe_unit_u)

    gt_c2w = _poses_from_pose_txt(gt_txt, fmt=gt_fmt)
    if gt_c2w is None:
        print(f"[evo_pose] 读取 GT 位姿失败: {gt_txt}")
        return {}

    # 帧数对齐
    S = min(len(pred_c2w), len(gt_c2w))
    pred_c2w = pred_c2w[:S]
    gt_c2w   = gt_c2w[:S]

    # 转 evo 轨迹
    traj_pred = _c2w_to_evo_traj(pred_c2w)
    traj_gt   = _c2w_to_evo_traj(gt_c2w)

    # -------- 基础轨迹图（纯 Matplotlib）--------
    try:


        gt_xyz   = traj_gt.positions_xyz
        pred_xyz = traj_pred.positions_xyz

        # 只用于绘图的对齐方式（可在 camera_cfg 里用 plot_align_mode 指定，默认 'sim3'）
        plot_align_mode = str(camera_cfg.get("plot_align_mode", "sim3")).lower()
        pred_xyz_vis = _align_pred_for_plot(gt_xyz, pred_xyz, mode=plot_align_mode)

        # XY 平面
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(gt_xyz[:, 0],   gt_xyz[:, 1],   '-', label='GT',   linewidth=1.5)
        ax.plot(pred_xyz_vis[:, 0], pred_xyz_vis[:, 1], '-', label='Pred', linewidth=1.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Trajectory (XY) [{plot_align_mode}]")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        fig.savefig(str(Path(out_dir) / "traj_xy.png"), dpi=200); plt.close(fig)

        if plot_3d:
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], '-', label='GT',   linewidth=1.2)
            ax3.plot(pred_xyz_vis[:, 0], pred_xyz_vis[:, 1], pred_xyz_vis[:, 2], '-', label='Pred', linewidth=1.2)
            ax3.set_title(f"Trajectory 3D [{plot_align_mode}]")
            ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
            ax3.legend()
            fig3.savefig(str(Path(out_dir) / "traj_3d.png"), dpi=200); plt.close(fig3)
    except Exception as e:
        print(f"[evo_pose] 轨迹绘图失败（可忽略）：{e}")

    # -------- 评估：多种对齐模式 --------
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    align_modes = _parse_align_modes(camera_cfg)

    # “尽量提取误差序列”的工具（不同版本结构不一致）
    def _extract_error_series(res_obj):
        for key in ("error_array", "errors", "error", "array", "values"):
            try:
                v = res_obj.np_arrays.get(key, None)   # 有些版本提供 np_arrays dict
                if v is not None:
                    return np.asarray(v).flatten()
            except Exception:
                pass
            try:
                v = getattr(res_obj, key, None)
                if v is not None:
                    v = np.asarray(v).flatten()
                    if v.size > 1:
                        return v
            except Exception:
                pass
        return None  # 实在没有就不画曲线

    for mode in align_modes:
        mode_tag = mode.lower()
        res_ape: Dict[str, float] = {}
        res_rpe: Dict[str, float] = {}

        # ===== APE: Trans / Rot =====
        try:
            ape_t = APE(PoseRelation.translation_part)
            _apply_align_settings(ape_t, mode_tag)
            ape_t.process_data((traj_gt, traj_pred))
            a_t = ape_t.get_result()

            ape_r = APE(PoseRelation.rotation_angle_deg)
            _apply_align_settings(ape_r, mode_tag)
            ape_r.process_data((traj_gt, traj_pred))
            a_r = ape_r.get_result()

            res_ape = {
                "trans_rmse":     float(a_t.stats["rmse"]),
                "trans_mean":     float(a_t.stats["mean"]),
                "trans_median":   float(a_t.stats["median"]),
                "trans_std":      float(a_t.stats["std"]),
                "rot_rmse_deg":   float(a_r.stats["rmse"]),
                "rot_mean_deg":   float(a_r.stats["mean"]),
                "rot_median_deg": float(a_r.stats["median"]),
                "rot_std_deg":    float(a_r.stats["std"]),
                "num_poses":      int(S),
                "mode":           mode_tag,
            }

            # 误差曲线（如可提取则绘）
            try:
                et = _extract_error_series(a_t)
                if et is not None and et.size > 1:
                    fig_et = plt.figure(); ax_et = fig_et.add_subplot(111)
                    ax_et.plot(et, '-', linewidth=1.2, label=f"APE-Trans ({mode_tag})")
                    ax_et.set_title(f"APE-Trans ({mode_tag})"); ax_et.set_xlabel("Frame"); ax_et.set_ylabel("Error")
                    ax_et.grid(True, linestyle='--', alpha=0.3); ax_et.legend()
                    fig_et.savefig(str(Path(out_dir) / f"ape_trans_curve_{mode_tag}.png"), dpi=200); plt.close(fig_et)

                er = _extract_error_series(a_r)
                if er is not None and er.size > 1:
                    fig_er = plt.figure(); ax_er = fig_er.add_subplot(111)
                    ax_er.plot(er, '-', linewidth=1.2, label=f"APE-Rot(deg) ({mode_tag})")
                    ax_er.set_title(f"APE-Rot(deg) ({mode_tag})"); ax_er.set_xlabel("Frame"); ax_er.set_ylabel("Error(deg)")
                    ax_er.grid(True, linestyle='--', alpha=0.3); ax_er.legend()
                    fig_er.savefig(str(Path(out_dir) / f"ape_rot_curve_{mode_tag}.png"), dpi=200); plt.close(fig_er)
            except Exception as e:
                print(f"[evo_pose] APE 曲线绘图失败（{mode_tag}）：{e}")

        except Exception as e:
            print(f"[evo_pose] APE 计算/绘图失败（{mode_tag}）：{e}")

        # ===== RPE: Trans / Rot =====
        try:
            rpe_t = RPE(PoseRelation.translation_part, delta=rpe_delta, delta_unit=rpe_unit_enum)
            _apply_align_settings(rpe_t, mode_tag)
            rpe_t.process_data((traj_gt, traj_pred))
            r_t = rpe_t.get_result()

            rpe_r = RPE(PoseRelation.rotation_angle_deg, delta=rpe_delta, delta_unit=rpe_unit_enum)
            _apply_align_settings(rpe_r, mode_tag)
            rpe_r.process_data((traj_gt, traj_pred))
            r_r = rpe_r.get_result()

            res_rpe = {
                "delta":          int(rpe_delta),
                "delta_unit":     "frames" if rpe_unit_enum == Unit.frames else "seconds",
                "trans_rmse":     float(r_t.stats["rmse"]),
                "trans_mean":     float(r_t.stats["mean"]),
                "trans_median":   float(r_t.stats["median"]),
                "trans_std":      float(r_t.stats["std"]),
                "rot_rmse_deg":   float(r_r.stats["rmse"]),
                "rot_mean_deg":   float(r_r.stats["mean"]),
                "rot_median_deg": float(r_r.stats["median"]),
                "rot_std_deg":    float(r_r.stats["std"]),
                "mode":           mode_tag,
            }

            # 误差曲线（如可提取则绘）
            try:
                et = _extract_error_series(r_t)
                if et is not None and et.size > 1:
                    fig_rt = plt.figure(); ax_rt = fig_rt.add_subplot(111)
                    ax_rt.plot(et, '-', linewidth=1.2, label=f"RPE-Trans Δ={rpe_delta} ({mode_tag})")
                    ax_rt.set_title(f"RPE-Trans Δ={rpe_delta} ({mode_tag})"); ax_rt.set_xlabel("Frame"); ax_rt.set_ylabel("Error")
                    ax_rt.grid(True, linestyle='--', alpha=0.3); ax_rt.legend()
                    fig_rt.savefig(str(Path(out_dir) / f"rpe_trans_curve_{mode_tag}.png"), dpi=200); plt.close(fig_rt)

                er = _extract_error_series(r_r)
                if er is not None and er.size > 1:
                    fig_rr = plt.figure(); ax_rr = fig_rr.add_subplot(111)
                    ax_rr.plot(er, '-', linewidth=1.2, label=f"RPE-Rot(deg) Δ={rpe_delta} ({mode_tag})")
                    ax_rr.set_title(f"RPE-Rot(deg) Δ={rpe_delta} ({mode_tag})"); ax_rr.set_xlabel("Frame"); ax_rr.set_ylabel("Error(deg)")
                    ax_rr.grid(True, linestyle='--', alpha=0.3); ax_rr.legend()
                    fig_rr.savefig(str(Path(out_dir) / f"rpe_rot_curve_{mode_tag}.png"), dpi=200); plt.close(fig_rr)
            except Exception as e:
                print(f"[evo_pose] RPE 曲线绘图失败（{mode_tag}）：{e}")

        except Exception as e:
            print(f"[evo_pose] RPE 计算/绘图失败（{mode_tag}）：{e}")

        results[mode_tag] = {"APE": res_ape, "RPE": res_rpe}

    # -------- 汇总 JSON --------
    out_path = Path(out_dir) / "metrics_camera_evo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
