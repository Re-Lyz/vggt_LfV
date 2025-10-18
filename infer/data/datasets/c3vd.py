import os.path as osp
import random
from glob import glob

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
    
from data.dataset_util import *
from data.base_dataset import BaseDataset


# =========================
# 通用序列容器
# =========================
@dataclass
class SequenceItem:
    images: List[str]
    gt: Dict[str, Any]
    meta: Dict[str, Any]


# =========================
# 简易预处理兜底
# =========================
def _stack_rgb_as_tensor(image_paths: List[str], device: str = "cpu"):
    arrs = []
    for p in image_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(im, (2, 0, 1))
        arrs.append(torch.from_numpy(chw))
    return torch.stack(arrs, 0).to(device)


class C3VDDatasetv1(BaseDataset):
    """
    读取 C3VD 的 registered 分支：
      - registered/<seq>/
          000x_depth.tiff
          000x_normals.tiff
          000x_occlusion.png
          x_color.png
        并在 <seq> 下有 pose.txt（每行16个数，行主序4x4，t在第4行前三个）

    目标：返回与 Co3dDataset.get_data 完全一致的 batch 结构。
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        ROOT: str = None,
        cameras=("cam0",),
        USE_REGISTERED: bool = True,
        min_num_images: int = 2,
        len_train: int = 100000,
        len_test: int = 10000,

        # 位姿方向：pose.txt 若是 c2w，则 assume_pose='c2w'；w2c 则 'w2c'
        assume_pose: str = "w2c",

        # 深度单位（如毫米→米）
        depth_unit_scale: float = 1.0,
    ):
        super().__init__(common_conf=common_conf)
        # 行为开关（与管线对齐）
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = getattr(common_conf, "allow_duplicate_img", True)

        assert ROOT is not None and osp.isdir(ROOT), f"Invalid ROOT: {ROOT}"
        self.ROOT = ROOT
        self.cameras = list(cameras)
        self.USE_REGISTERED = bool(USE_REGISTERED)

        self.len_train = len_train if split == "train" else len_test

        self.assume_pose = assume_pose
        self.depth_unit_scale = float(depth_unit_scale)

        # —— 从 common_conf 读取 (H, W)、OCam 与兜底 FOV
        self.H, self.W = get_hw_from_common_conf(common_conf)
        self.use_ocam_as_pinhole = getattr(common_conf, "use_ocam_as_pinhole", True)
        self.fov_fallback_deg = float(getattr(common_conf, "fov_fallback_deg", 60.0))
        
        self.K = None
        if self.use_ocam_as_pinhole:
            try:
                self.K = ocam_to_pinhole_K_from_cfg(common_conf, (self.H, self.W), zero_skew=True)
            except Exception as ex:
                if self.debug:
                    print(f"[OCam->K] failed at init: {ex}. Fallback to FOV.")

        if self.K is None:
            f = (max(self.H, self.W) * 0.5) / np.tan(np.deg2rad(self.fov_fallback_deg * 0.5))
            self.K = np.array([[f, 0, self.W * 0.5],
                               [0, f, self.H * 0.5],
                               [0, 0, 1.0]], dtype=np.float64)
            
        # —— 扫描数据，填充 data_store 与 sequence_list
        self.data_store = {}      # seq_name -> list[ per-frame dict ]
        self.sequence_list = []

        seq_dirs = sorted([d for d in glob(osp.join(self.ROOT, "*")) if osp.isdir(d)])
        # print("seq_dirs:", seq_dirs)
        for seq_dir in seq_dirs:
            seq_name = osp.basename(seq_dir)
            pose_txt = osp.join(seq_dir, "pose.txt")
            if not osp.isfile(pose_txt):
                if self.debug:
                    print(f"[Skip] pose.txt not found in {seq_dir}")
                continue
            
            with open(pose_txt, "r") as f:
                pose_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            # registered 子目录若存在则用；否则就用序列根目录
            reg_dir = osp.join(seq_dir, "registered")
            if self.USE_REGISTERED and osp.isdir(reg_dir):
                data_dir = reg_dir
            else:
                data_dir = seq_dir

            # 列出颜色帧
            color_files = sorted(glob(osp.join(data_dir, "*_color.png")))
            if self.debug:
                print(f"[Info] Found {len(color_files)} color images in {data_dir}")
            if len(color_files) == 0:
                if self.debug:
                    print(f"[Warn] No *_color.png in {data_dir}")
                continue
            
            seq_items = []
            for cf in color_files:
                bn = osp.basename(cf)
                # 兼容 "3_color.png" / "0003_color.png"
                idx_str = bn.split("_")[0]
                try:
                    idx = int(idx_str)
                except Exception:
                    idx = int(idx_str.lstrip("0") or "0")

                # pose 索引自适应：优先 0-based；不够就试 1-based
                pose_idx = idx
                if pose_idx >= len(pose_lines) and (idx - 1) >= 0 and (idx - 1) < len(pose_lines):
                    pose_idx = idx - 1
                if pose_idx < 0 or pose_idx >= len(pose_lines):
                    if self.debug:
                        print(f"[Skip] pose index out of range: seq={seq_name}, frame={idx}, poses={len(pose_lines)}")
                    continue
                
                extri_w2c = parse_pose_row_major_4x4(pose_lines[pose_idx], assume=self.assume_pose)
                if self.debug:
                    print(f"  frame {idx:04d}: pose line {pose_idx}, extri_w2c=\n{extri_w2c}")
                    print()
                
                idx4 = f"{idx:04d}"
                depth_path   = osp.join(data_dir, f"{idx4}_depth.tiff")
                normals_path = osp.join(data_dir, f"{idx4}_normals.tiff")
                occ_path     = osp.join(data_dir, f"{idx4}_occlusion.png")

                seq_items.append(dict(
                    frame_idx=idx,
                    color_path=cf,
                    depth_path=depth_path,
                    normals_path=normals_path,
                    occ_path=occ_path,
                    extri_w2c=extri_w2c,
                    K=self.K,   # 整个序列用一套 K（来自 common_conf）
                    H=self.H, W=self.W,
                    cam_name=None,  # 无多相机
                ))
                if self.debug:
                    print(f"  Added frame {idx:04d}: color={cf}, depth={depth_path}, occ={occ_path}")

            if len(seq_items) >= min_num_images:
                seq_items.sort(key=lambda x: x["frame_idx"])
                self.data_store[seq_name] = seq_items
                self.sequence_list.append(seq_name)

        self.sequence_list_len = len(self.sequence_list)
        if self.debug:
            print(f"[C3VDDatasetv1] {split}: {self.sequence_list_len} sequences loaded from {self.ROOT}")
            for sn in self.sequence_list:
                print(f"  seq '{sn}': {len(self.data_store[sn])} images")

    # ————————————————————————————————————————————————————————————————————————

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 4,
        seq_name: str = None,
        ids: list | None = None,
        aspect_ratio: float = 1.0,
    ):
        """
        返回与 Co3dDataset.get_data 完全一致的 batch。
        """
        assert self.sequence_list_len > 0, "Empty C3VD dataset after scanning."

        if self.inside_random or (seq_index is None):
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        meta = self.data_store[seq_name]
        if self.debug:
            print("meta:", len(meta))
            
        if ids is None:
            ids = np.random.choice(len(meta), img_per_seq, replace=self.allow_duplicate_img)
        if self.debug:
            print("ids:", ids)
            
        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            anno = meta[int(i)]
            color_path = anno["color_path"]
            # if self.debug:
            #     print(f"Processing frame {i}: {color_path}")

            # 1) 读图（图像本身仍按实际尺寸加载；K 已按 common_conf 的 H,W 生成）
            image = read_image_cv2(color_path)
            # if self.debug:
                # print(f"  Read image: {color_path}, shape: {None if image is None else image.shape}")

            if image is None:
                continue

            # 2) 读深度（可选）并处理 occlusion
            if self.load_depth and osp.isfile(anno["depth_path"]):
                depth_map = read_depth_any(anno["depth_path"], self.depth_unit_scale)
                if osp.isfile(anno["occ_path"]):
                    occ = cv2.imread(anno["occ_path"], cv2.IMREAD_GRAYSCALE)
                    if occ is not None:
                        depth_map[occ > 0] = 0.0
                depth_map = threshold_depth_map(depth_map, max_percentile=98, min_percentile=-1)
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])

            # 3) 位姿/内参（OpenCV 约定；外参为 w->c 3x4）
            extri_opencv = np.array(anno["extri_w2c"], dtype=np.float64)
            intri_opencv = np.array(anno["K"], dtype=np.float64)

            # 4) 统一预处理与点云
            (image, depth_map, extri_opencv, intri_opencv,
             world_coords_points, cam_coords_points, point_mask, _) = self.process_one_image(
                image=image,
                depth_map=depth_map,
                extri_opencv=extri_opencv,
                intri_opencv=intri_opencv,
                original_size=original_size,
                target_image_shape=target_image_shape,
                filepath=color_path
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            world_points.append(world_coords_points)
            cam_points.append(cam_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        # ids 输出为 list，兼容序列化
        ids_out = ids.tolist() if isinstance(ids, np.ndarray) else list(ids)

        batch = {
            "seq_name": "c3vd_" + seq_name,
            "ids": ids_out,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        if getattr(self, "debug", False):
            batch["_debug_image_paths"] = [meta[int(i)]["color_path"] for i in ids_out]
            batch["_debug_depth_paths"] = [
                meta[int(i)]["depth_path"] if osp.isfile(meta[int(i)]["depth_path"]) else None
                for i in ids_out
            ]
            batch["_debug_occ_paths"] = [
                meta[int(i)]["occ_path"] if osp.isfile(meta[int(i)]["occ_path"]) else None
                for i in ids_out
            ]

        return batch
    
    

class C3VDDatasetV1Adapter:
    """
    目标接口保持不变：
      - list_sequences() -> List[str]                 # 返回“窗口化”的序列ID
      - build_sequence(seq_id) -> SequenceItem        # 由窗口ID构造子序列的 paths/gt/meta
      - load_and_preprocess_images(paths, device) -> Tensor[B,3,H,W]
      - load_and_preprocess_query_points(...) -> None

    主要变化：
      1) 把 base.data_store[seq] 的帧切成窗口（长度 L，步长 stride 或随机起点）
      2) seq_id 形如： "seqName|12:60" 表示 [start=12, end=60) 的窗口
      3) 窗口长度 L 的优先级： common_config.fix_img_num(>0) > ds_cfg.max_img_per_gpu > 48
      4) inside_random=True 时，窗口内还可随机抽取 L 帧；否则按顺序
      5) allow_duplicate_img 控制短尾或不足 L 时的补齐策略
    """

    def __init__(self, base: C3VDDatasetv1, ds_cfg: dict):
        self.base = base
        self.ds_cfg = ds_cfg or {}
        self.ROOT = getattr(base, "ROOT", None)

        # ===== 从 common_config 读取“训练同款”的行为开关 =====
        cc = getattr(base, "common_conf", None)
        # L（窗口长度）
        fix_img_num = getattr(cc, "fix_img_num", -1) if cc else -1
        max_img_per_gpu = int(self.ds_cfg.get("max_img_per_gpu", 48))
        self.L = int(fix_img_num) if isinstance(fix_img_num, int) and fix_img_num > 0 else int(max_img_per_gpu)
        if self.L <= 0:
            self.L = 48

        # 取样行为
        self.inside_random = bool(getattr(cc, "inside_random", True)) if cc else bool(self.ds_cfg.get("inside_random", True))
        self.allow_duplicate_img = bool(getattr(cc, "allow_duplicate_img", True)) if cc else bool(self.ds_cfg.get("allow_duplicate_img", True))

        # 滑窗配置：不设则每条序列仅给一个窗口（随机 / 头对齐）
        slide_stride = getattr(cc, "slide_stride", None) if cc else self.ds_cfg.get("slide_stride", None)
        self.slide_stride = None if slide_stride in (None, "null") else int(slide_stride)
        self.drop_last = bool(getattr(cc, "drop_last", False)) if cc else bool(self.ds_cfg.get("drop_last", False))

        # 评估复现：推理默认固定随机性
        self.eval_mode = True
        self._rng = np.random.RandomState(0) if self.eval_mode else np.random

        # 原始序列列表
        self._seq_names: List[str] = list(getattr(self.base, "sequence_list", []))

        # === 预构建窗口索引 ===
        # 形式： self._windows = List[ (seq_name, start, end) ]
        self._windows: List[tuple[str, int, int]] = []
        self._seq2len: Dict[str, int] = {}
        self._build_windows()

    # ----------------- 公共接口 -----------------

    def list_sequences(self) -> List[str]:
        """
        返回窗口化后的“序列ID”列表，每个元素形如 "seqName|start:end"。
        如果需要限制总窗口数，可以在 ds_cfg 里设 limit_windows（可选）。
        """
        ids = [self._encode_wid(sn, s, e) for (sn, s, e) in self._windows]
        limit = int(self.ds_cfg.get("limit_windows", -1))
        return ids[:limit] if limit > 0 else ids

    def build_sequence(self, seq_id: str) -> SequenceItem:
        """
        根据窗口ID（seqName|start:end）构造 SequenceItem（只含该窗口内的帧）。
        """
        seq_name, start, end = self._decode_wid(seq_id)
        if seq_name not in self.base.data_store:
            raise KeyError(f"[C3VDDatasetV1Adapter] sequence not found: {seq_name}")
        frames_all = self.base.data_store[seq_name]
        n = len(frames_all)

        # 真实索引（含窗口内的二次采样与补齐）
        idxs = self._make_indices(start, end, n)

        frames = [frames_all[i] for i in idxs]

        images = [f["color_path"] for f in frames]
        depth_paths   = [f["depth_path"]   for f in frames]
        mask_paths    = [f["occ_path"]     for f in frames]
        normals_paths = [f["normals_path"] for f in frames]

        # data_dir：取第一帧所在目录
        from os.path import dirname
        data_dir = dirname(images[0]) if images else (self.ROOT and str(Path(self.ROOT, seq_name)))

        gt = {
            "depth_paths": depth_paths,
            "valid_masks": mask_paths,
            "normals_paths": normals_paths,
            # 如需可额外传：intrinsics / extrinsics
            # "intrinsics": [f["K"] for f in frames],
            # "extrinsics_w2c": [f["extri_w2c"] for f in frames],
            "global_ids": [f["frame_idx"] for f in frames],  # 回拼评估时有用
        }
        meta = {
            "sequence_name": seq_name,
            "sequence_dir": str(Path(self.ROOT, seq_name)) if self.ROOT else None,
            "data_dir": data_dir,
            "pose_txt": str(Path(self.ROOT, seq_name, "pose.txt")) if self.ROOT else None,
            "window_range": (int(start), int(end)),
            "window_length": int(self.L),
        }
        return SequenceItem(images=images, gt=gt, meta=meta)

    def load_and_preprocess_images(self, image_paths: List[str], device: str) -> torch.Tensor:
        # 1) 若 base 提供了更合适的预处理，优先用
        if hasattr(self.base, "load_and_preprocess_images") and callable(self.base.load_and_preprocess_images):
            try:
                return self.base.load_and_preprocess_images(image_paths, device=device)
            except TypeError:
                return self.base.load_and_preprocess_images(image_paths).to(device)
            except Exception:
                pass
        # 2) VGGT 官方兜底
        try:
            from vggt.utils.load_fn import load_and_preprocess_images as official
            return official(image_paths).to(device)
        except Exception:
            pass
        # 3) 最终兜底
        return _stack_rgb_as_tensor(image_paths, device=device)

    def load_and_preprocess_query_points(self, seq_item, device: str):
        return None

    # ----------------- 内部方法 -----------------

    def _build_windows(self):
        self._windows.clear()
        self._seq2len.clear()

        for sn in self._seq_names:
            frames = self.base.data_store.get(sn, [])
            n = len(frames)
            self._seq2len[sn] = n
            if n <= 0:
                continue

            if self.slide_stride is not None and self.slide_stride > 0:
                # —— 滑动窗口 —— 与训练中 shuffle=False、inside_random=False 的稳定行为相似
                stride = int(self.slide_stride)
                starts = list(range(0, max(1, n - self.L + 1), stride))
                for s in starts:
                    e = s + self.L
                    if e <= n:
                        self._windows.append((sn, s, e))
                # 尾窗（右对齐）：确保覆盖到序列末尾
                if not self.drop_last and (n < self.L or (n - self.L) % stride != 0):
                    s = max(0, n - self.L)
                    e = s + self.L
                    if (sn, s, e) not in self._windows:
                        self._windows.append((sn, s, e))
            else:
                # —— 随机选择一个窗口（推理通常每序列一个就够）——
                if n >= self.L:
                    if self.inside_random:
                        s = int(self._rng.randint(0, n - self.L + 1))
                    else:
                        s = 0
                    e = s + self.L
                    self._windows.append((sn, s, e))
                else:
                    # 过短也给一个窗口（0, n），后续 _make_indices 里再补齐到 L
                    self._windows.append((sn, 0, n))

        # 可选：限制每条序列的窗口数量
        limit_per_seq = int(self.ds_cfg.get("limit_windows_per_seq", -1))
        if limit_per_seq > 0 and self.slide_stride is not None:
            filtered = []
            seen = {}
            for sn, s, e in self._windows:
                cnt = seen.get(sn, 0)
                if cnt < limit_per_seq:
                    filtered.append((sn, s, e))
                    seen[sn] = cnt + 1
            self._windows = filtered

    def _make_indices(self, start: int, end: int, n: int) -> List[int]:
        """
        从 [start, end) 生成长度为 L 的帧索引：
         - 若 end-start >= L：
              inside_random=True  ⇒ 在窗口内随机选 L 帧并排序
              inside_random=False ⇒ 直接取 start..start+L-1
         - 若不足 L：按 allow_duplicate_img 规则补齐
        """
        start = int(start); end = int(end)
        length = max(0, min(end, n) - max(0, start))

        if length >= self.L:
            if self.inside_random:
                chosen = sorted(self._rng.choice(np.arange(start, start + length), size=self.L, replace=False))
                return [int(i) for i in chosen]
            else:
                return list(range(start, start + self.L))

        # 不足 L：补齐
        base_idxs = list(range(start, start + length))
        if length <= 0:
            # 极端情况：序列空或窗口越界，退化为重复最后一帧 0
            fill_val = min(max(n - 1, 0), n - 1) if n > 0 else 0
            return [fill_val] * self.L

        if self.allow_duplicate_img:
            # 重复最后一帧到 L
            pad = [base_idxs[-1]] * (self.L - length)
            return base_idxs + pad
        else:
            # 循环补齐
            idxs = []
            while len(idxs) < self.L:
                take = min(length, self.L - len(idxs))
                idxs.extend(base_idxs[:take])
            return idxs

    @staticmethod
    def _encode_wid(seq_name: str, start: int, end: int) -> str:
        return f"{seq_name}|{int(start)}:{int(end)}"

    @staticmethod
    def _decode_wid(seq_id: str) -> tuple[str, int, int]:
        # 兼容纯序列名（无窗口参数）的旧用法：退化为整段
        if "|" not in seq_id:
            name = seq_id
            start, end = 0, 1 << 30
            return name, start, end
        name, rng = seq_id.split("|", 1)
        s, e = rng.split(":")
        return name, int(s), int(e)
    
    