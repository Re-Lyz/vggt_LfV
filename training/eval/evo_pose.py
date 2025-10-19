# eval/evo_pose.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import json, os
import torch
import matplotlib.pyplot as plt
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from evo.core.metrics import  Unit
    EVO_AVAILABLE = True
    _EVO_IMPORT_ERROR = None
except Exception as e:
    EVO_AVAILABLE = False
    _EVO_IMPORT_ERROR = e


# ---------- helpers ----------
def _to_np(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _quat_to_R(q, order: str = "xyzw"):
    """q: [...,4]; order: 'xyzw' or 'wxyz'"""
    q = np.asarray(q, dtype=np.float64)
    if order.lower() == "wxyz":
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:  # 默认 xyzw
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.maximum(np.sqrt(x*x + y*y + z*z + w*w), 1e-12)
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.stack([
        1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx),
        2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R

def _poses_from_pose_enc(pose_enc: np.ndarray, quat_order: str = "xyzw") -> np.ndarray:
    """
    absT_quaR_FoV 编码 -> c2w
    输入: pose_enc [N,D] 或 [B,T,D]，切片：[0:3]=T, [3:7]=quat
    输出: c2w [N,4,4]
    """
    P = _to_np(pose_enc)
    if P.ndim == 3:
        P = P.reshape(-1, P.shape[-1])  # 合并 BT
    assert P.ndim == 2 and P.shape[1] >= 7, f"pose_enc 形状需为 [*,>=7]，当前 {P.shape}"
    T = P[:, :3]
    Q = P[:, 3:7]
    Rm = _quat_to_R(Q, order=quat_order)
    N = P.shape[0]
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None, ...], N, axis=0)
    c2w[:, :3, :3] = Rm
    c2w[:, :3,  3] = T
    return c2w

def _c2w_to_evo_traj(c2w: np.ndarray):
    from evo.core.trajectory import PoseTrajectory3D
    S  = c2w.shape[0]
    xyz = c2w[:, :3, 3]
    Rm  = c2w[:, :3, :3]
    try:
        from scipy.spatial.transform import Rotation as R_
        q_xyzw = R_.from_matrix(Rm).as_quat()  # (x,y,z,w)
        q_wxyz = np.concatenate([q_xyzw[:, -1:], q_xyzw[:, :3]], axis=1)
    except Exception:
        from evo.core import transformations as tf
        q_wxyz = []
        for i in range(S):
            M = np.eye(4); M[:3,:3] = Rm[i]
            q_wxyz.append(tf.quaternion_from_matrix(M))
        q_wxyz = np.asarray(q_wxyz)
    stamps = np.arange(S, dtype=float)
    return PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=q_wxyz, timestamps=stamps)

def _parse_align_modes(camera_cfg: Dict[str, Any]) -> List[str]:
    a = camera_cfg.get("align", "sim3")
    if isinstance(a, str): return [a.lower()]
    if isinstance(a, (list, tuple)): return [str(x).lower() for x in a]
    return ["sim3"]

def _apply_align_settings(metric, mode: str):
    mode = (mode or "").lower()
    if   mode == "none": align, correct_scale = False, False
    elif mode == "se3":  align, correct_scale = True,  False
    elif mode == "sim3": align, correct_scale = True,  True
    elif mode == "scale":align, correct_scale = False, True
    else:                align, correct_scale = True,  True
    if hasattr(metric, "settings") and metric.settings is not None:
        try:
            metric.settings.align = align
            metric.settings.correct_scale = correct_scale
            return
        except Exception:
            pass
    if hasattr(metric, "align"):
        try: metric.align = align
        except Exception: pass
    if hasattr(metric, "correct_scale"):
        try: metric.correct_scale = correct_scale
        except Exception: pass

def _align_pred_for_plot(gt_xyz: np.ndarray, pred_xyz: np.ndarray, mode: str = "sim3"):
    def _umeyama_sim3(X, Y, with_scale=True):
        X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
        muX, muY = X.mean(0), Y.mean(0)
        Xc, Yc = X - muX, Y - muY
        cov = (Yc.T @ Xc) / X.shape[0]
        U, S, Vt = np.linalg.svd(cov)
        C = np.ones(3)
        if np.linalg.det(U @ Vt) < 0: C[-1] = -1
        R = U @ np.diag(C) @ Vt
        if with_scale:
            varY = (Yc*Yc).sum()/X.shape[0]
            s = (S*C).sum() / max(varY, 1e-12)
        else:
            s = 1.0
        t = muX - s*(R@muY)
        return float(s), R, t

    mode = (mode or "sim3").lower()
    if pred_xyz.shape[0] < 3 or gt_xyz.shape[0] < 3:
        return pred_xyz
    try:
        if mode == "sim3":
            s,R,t = _umeyama_sim3(gt_xyz,pred_xyz,with_scale=True);  return (s*(pred_xyz@R.T)+t)
        if mode == "se3":
            s,R,t = _umeyama_sim3(gt_xyz,pred_xyz,with_scale=False); return (pred_xyz@R.T)+t
        if mode == "scale":
            eps=1e-12; lg=np.linalg.norm(gt_xyz[-1]-gt_xyz[0])+eps; lp=np.linalg.norm(pred_xyz[-1]-pred_xyz[0])+eps
            s = lg/lp;  return (pred_xyz - pred_xyz.mean(0))*s + gt_xyz.mean(0)
        return pred_xyz
    except Exception:
        return pred_xyz

def _extract_error_series(res_obj):
    # 兼容不同 evo 版本
    for key in ("np_arrays","error_array","errors","error","array","values"):
        try:
            if key == "np_arrays" and hasattr(res_obj, "np_arrays"):
                d = getattr(res_obj, "np_arrays")
                for k2 in ("error_array","errors","error","array","values"):
                    v = d.get(k2, None)
                    if v is not None:
                        v = np.asarray(v).flatten()
                        if v.size > 1: return v
            else:
                v = getattr(res_obj, key, None)
                if v is not None:
                    v = np.asarray(v).flatten()
                    if v.size > 1: return v
        except Exception:
            pass
    return None


# ---------- 主评估接口：直接吃 pred_pose_enc_valid / gt_pose_enc_valid ----------
def evaluate_and_visualize_poses(
    pred_pose_enc_valid,   # [N,D] 或 [B,T,D]（已按有效帧筛好）
    gt_pose_enc_valid,     # 同上
    out_dir: Path,
    camera_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    用 evo 评估位姿（不再读取 .pt，不再从 dict 中拿）：
      - 输入为已筛过帧的 absT_quaR_FoV 编码：前 3 = T，后 4 = quat
      - camera_cfg 可设置:
        * quat_order: 'xyzw' | 'wxyz'（默认 'xyzw'）
        * align: 'none' | 'se3' | 'sim3' | 'scale' 或列表
        * plot_align_mode: 可视化对齐方式（不影响指标），默认 'sim3'
        * plot_3d: 是否画 3D 轨迹（默认 True）
        * rpe_delta / rpe_delta_unit: RPE 设置
    """
    if not EVO_AVAILABLE:
        print(f"[evo_pose] 未安装 evo，跳过相机评估：{_EVO_IMPORT_ERROR}")
        return {}

    quat_order   = str(camera_cfg.get("quat_order", "xyzw")).lower()
    plot_3d      = bool(camera_cfg.get("plot_3d", True))
    rpe_delta    = int(camera_cfg.get("rpe_delta", 1))
    unit_str     = str(camera_cfg.get("rpe_delta_unit", "frames")).lower()
    rpe_unit_enum = Unit.frames if unit_str in ("frames","frame","f","index","indices","idx","samples") else Unit.seconds

    # 编码 -> c2w
    pred_c2w = _poses_from_pose_enc(pred_pose_enc_valid, quat_order)  # [N,4,4]
    gt_c2w   = _poses_from_pose_enc(gt_pose_enc_valid,   quat_order)

    # 帧数对齐
    S = min(len(pred_c2w), len(gt_c2w))
    if S < 2:
        print("[evo_pose] 可用帧数不足（<2），跳过。")
        return {}
    pred_c2w = pred_c2w[:S]
    gt_c2w   = gt_c2w[:S]

    # 转 evo 轨迹
    from evo.core.metrics import APE, RPE, PoseRelation
    traj_pred = _c2w_to_evo_traj(pred_c2w)
    traj_gt   = _c2w_to_evo_traj(gt_c2w)

    # ---- 可视化：XY/3D ----
    try:
        gt_xyz   = traj_gt.positions_xyz
        pred_xyz = traj_pred.positions_xyz
        plot_align_mode = str(camera_cfg.get("plot_align_mode", "sim3")).lower()
        pred_xyz_vis = _align_pred_for_plot(gt_xyz, pred_xyz, mode=plot_align_mode)

        # XY
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(gt_xyz[:,0], gt_xyz[:,1], '-', label='GT', linewidth=1.5)
        ax.plot(pred_xyz_vis[:,0], pred_xyz_vis[:,1], '-', label='Pred', linewidth=1.5)
        ax.set_aspect('equal', adjustable='box'); ax.set_title(f"Trajectory (XY) [{plot_align_mode}]")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, linestyle='--', alpha=0.3); ax.legend()
        fig.savefig(str(Path(out_dir) / "traj_xy.png"), dpi=200); plt.close(fig)

        if plot_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig3 = plt.figure(); ax3 = fig3.add_subplot(111, projection='3d')
            ax3.plot(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], '-', label='GT', linewidth=1.2)
            ax3.plot(pred_xyz_vis[:,0], pred_xyz_vis[:,1], pred_xyz_vis[:,2], '-', label='Pred', linewidth=1.2)
            ax3.set_title(f"Trajectory 3D [{plot_align_mode}]")
            ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z"); ax3.legend()
            fig3.savefig(str(Path(out_dir) / "traj_3d.png"), dpi=200); plt.close(fig3)
    except Exception as e:
        print(f"[evo_pose] 轨迹绘图失败（可忽略）：{e}")

    # ---- 评估：多对齐模式（APE + RPE）----
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    align_modes = _parse_align_modes(camera_cfg)

    for mode in align_modes:
        mode_tag = mode.lower()
        res_ape: Dict[str, float] = {}
        res_rpe: Dict[str, float] = {}

        # APE
        try:
            ape_t = APE(PoseRelation.translation_part)
            _apply_align_settings(ape_t, mode_tag); ape_t.process_data((traj_gt, traj_pred)); a_t = ape_t.get_result()

            ape_r = APE(PoseRelation.rotation_angle_deg)
            _apply_align_settings(ape_r, mode_tag); ape_r.process_data((traj_gt, traj_pred)); a_r = ape_r.get_result()

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

            # 曲线（若能提取）
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

        # RPE
        try:
            rpe_t = RPE(PoseRelation.translation_part, delta=rpe_delta, delta_unit=rpe_unit_enum)
            _apply_align_settings(rpe_t, mode_tag); rpe_t.process_data((traj_gt, traj_pred)); r_t = rpe_t.get_result()

            rpe_r = RPE(PoseRelation.rotation_angle_deg, delta=rpe_delta, delta_unit=rpe_unit_enum)
            _apply_align_settings(rpe_r, mode_tag); rpe_r.process_data((traj_gt, traj_pred)); r_r = rpe_r.get_result()

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

            # 曲线
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

    # 汇总
    out_path = Path(out_dir) / "metrics_camera_evo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results
