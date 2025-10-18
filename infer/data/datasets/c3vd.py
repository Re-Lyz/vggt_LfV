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
    多序列 + 分层均匀抽样（无窗口）
    - list_sequences(): 返回所有序列名
    - build_sequence(seq_id): 对该序列均匀抽 K 张（不够全取；可选补齐到 K）
    """
    def __init__(self, base: C3VDDatasetv1, ds_cfg: dict):
        import numpy as np
        self.base = base
        self.ds_cfg = ds_cfg or {}
        self.ROOT = getattr(base, "ROOT", None)

        strat = (self.ds_cfg.get("stratified_eval") or {})
        self.k_per_seq   = int(strat.get("num_samples_per_seq", 32))  # 每序列抽样张数 K
        self.jitter      = bool(strat.get("jitter", False))           # 桶内随机
        self.seed        = int(strat.get("seed", 0))                  # 随机种子
        self.pad_to_k    = bool(strat.get("pad_to_k", False))         # 不足是否补齐到 K
        self.pad_mode    = str(strat.get("pad_mode", "repeat"))       # repeat / loop

        self._rng = np.random.RandomState(self.seed)
        self._seq_names = list(getattr(self.base, "sequence_list", []))
        self._seq2len   = {sn: len(self.base.data_store.get(sn, [])) for sn in self._seq_names}

    # ---------- 公共接口 ----------
    def list_sequences(self) -> List[str]:
        limit = int(self.ds_cfg.get("limit_seqs", -1) or -1)
        seqs = self._seq_names
        return seqs[:limit] if limit > 0 else seqs

    def build_sequence(self, seq_id: str) -> SequenceItem:
        import numpy as np
        if seq_id not in self.base.data_store:
            raise KeyError(f"[C3VDDatasetV1Adapter] sequence not found: {seq_id}")

        frames_all = self.base.data_store[seq_id]
        n = len(frames_all)

        # 分层均匀抽样
        K = max(1, int(self.k_per_seq))
        idxs = self._stratified_pick(n, K, jitter=self.jitter, rng=self._rng)

        # 不足是否补齐到 K（用于模型前向需要定长的情况）
        if self.pad_to_k and len(idxs) < K and n > 0:
            need = K - len(idxs)
            if self.pad_mode == "loop":
                more = []
                i = 0
                while len(more) < need:
                    more.append(idxs[i % len(idxs)])
                    i += 1
            else:  # repeat
                more = [idxs[-1]] * need
            idxs = idxs + more

        frames = [frames_all[i] for i in idxs]
        images = [f["color_path"] for f in frames]
        depth_paths   = [f["depth_path"]   for f in frames]
        mask_paths    = [f["occ_path"]     for f in frames]
        normals_paths = [f["normals_path"] for f in frames]

        from os.path import dirname
        data_dir = dirname(images[0]) if images else (self.ROOT and str(Path(self.ROOT, seq_id)))

        gt = {
            "depth_paths": depth_paths,
            "valid_masks": mask_paths,
            "normals_paths": normals_paths,
            "global_ids": [f["frame_idx"] for f in frames],
        }
        meta = {
            "sequence_name": seq_id,
            "sequence_dir": str(Path(self.ROOT, seq_id)) if self.ROOT else None,
            "data_dir": data_dir,
            "pose_txt": str(Path(self.ROOT, seq_id, "pose.txt")) if self.ROOT else None,
            "sampled_length": len(images),
            "sampling": {
                "strategy": "stratified_uniform",
                "num_samples_per_seq": self.k_per_seq,
                "jitter": self.jitter,
                "seed": self.seed,
                "pad_to_k": self.pad_to_k,
                "pad_mode": self.pad_mode,
            },
        }
        return SequenceItem(images=images, gt=gt, meta=meta)

    def load_and_preprocess_images(self, image_paths: List[str], device: str) -> torch.Tensor:
        if hasattr(self.base, "load_and_preprocess_images") and callable(self.base.load_and_preprocess_images):
            try:
                return self.base.load_and_preprocess_images(image_paths, device=device)
            except TypeError:
                return self.base.load_and_preprocess_images(image_paths).to(device)
            except Exception:
                pass
        try:
            from vggt.utils.load_fn import load_and_preprocess_images as official
            return official(image_paths).to(device)
        except Exception:
            pass
        return _stack_rgb_as_tensor(image_paths, device=device)

    def load_and_preprocess_query_points(self, seq_item, device: str):
        return None

    # ---------- 内部 ----------
    @staticmethod
    def _stratified_pick(n: int, k: int, jitter: bool, rng) -> List[int]:
        import numpy as np
        if n <= 0:
            return []
        if k >= n:
            return list(range(n))  # 不够就全取
        edges = np.linspace(0, n, num=k+1, endpoint=True, dtype=float)
        idxs = []
        for i in range(k):
            a, b = edges[i], edges[i+1]
            if jitter:
                x = rng.uniform(a, max(a, b - 1e-6))
                j = int(np.clip(np.floor(x), 0, n-1))
            else:
                x = (a + b) * 0.5
                j = int(np.clip(int(round(x - 0.5)), 0, n-1))
            idxs.append(j)
        idxs = sorted(set(idxs))
        while len(idxs) < min(k, n):
            gaps = [idxs[i+1]-idxs[i] for i in range(len(idxs)-1)]
            if not gaps:
                break
            pos = int(np.argmax(gaps))
            cand = (idxs[pos] + idxs[pos+1]) // 2
            if cand not in idxs:
                idxs.insert(pos+1, int(cand))
            else:
                break
        return idxs[:min(k, n)]

    