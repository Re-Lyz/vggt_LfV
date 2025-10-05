import os
import os.path as osp
import random
from glob import glob

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


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

            if len(seq_items) >= min_num_images:
                seq_items.sort(key=lambda x: x["frame_idx"])
                self.data_store[seq_name] = seq_items
                self.sequence_list.append(seq_name)

        self.sequence_list_len = len(self.sequence_list)

    # ————————————————————————————————————————————————————————————————————————

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 2,
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
        if ids is None:
            ids = np.random.choice(len(meta), img_per_seq, replace=self.allow_duplicate_img)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            anno = meta[int(i)]
            color_path = anno["color_path"]

            # 1) 读图（图像本身仍按实际尺寸加载；K 已按 common_conf 的 H,W 生成）
            image = read_image_cv2(color_path)
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