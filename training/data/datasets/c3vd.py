import os
import os.path as osp
import random
from glob import glob

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


# —— 若想更稳妥地做 SE(3) 求逆（和你工具里一致）：
from vggt.utils.geometry import closed_form_inverse_se3


class C3VDDatasetv1(BaseDataset):
    """
    读取 C3VD 的 registered 分支：
      - <ROOT>/<seq>/<camera>/registered/
          000x_depth.tiff
          000x_normals.tiff
          000x_occlusion.png
          x_color.png
        并在 <camera> 下有 pose.txt（每行16个数，行主序4x4，t在第4行前三个）

    目标：返回与 Co3dDataset.get_data 完全一致的 batch 结构。
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        ROOT: str = None,
        cameras=("cam0",),          # 你数据中的相机子目录名
        USE_REGISTERED: bool = True,
        min_num_images: int = 2,
        len_train: int = 100000,
        len_test: int = 10000,

        # —— 内参（若数据未提供）：
        default_fx: float | None = None,
        default_fy: float | None = None,
        default_cx: float | None = None,
        default_cy: float | None = None,
        default_fov_deg: float | None = 60.0,  # 无 fx/fy 时用 FOV 估计

        # —— 位姿方向：
        # 若 pose.txt 实际是 c2w，请把 assume_pose='c2w'；如果是 w2c，保持 'w2c'
        assume_pose: str = "w2c",

        # —— 深度单位（如毫米→米）
        depth_unit_scale: float = 1.0,
    ):
        super().__init__(common_conf=common_conf)

        # 保持与现有管线一致的行为开关
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

        # 内参默认（若帧级/相机级没有提供）
        self.default_fx = default_fx
        self.default_fy = default_fy
        self.default_cx = default_cx
        self.default_cy = default_cy
        self.default_fov_deg = default_fov_deg

        self.assume_pose = assume_pose
        self.depth_unit_scale = float(depth_unit_scale)

        # —— 扫描数据，填充 data_store 与 sequence_list
        self.data_store = {}      # seq_name -> list[ per-frame dict ]
        self.sequence_list = []

        seq_dirs = sorted([d for d in glob(osp.join(self.ROOT, "*")) if osp.isdir(d)])
        for seq_dir in seq_dirs:
            seq_name = osp.basename(seq_dir)
            seq_items = []

            for cam in self.cameras:
                cam_dir = osp.join(seq_dir, cam)
                if not osp.isdir(cam_dir):
                    continue

                pose_txt = osp.join(cam_dir, "pose.txt")
                if not osp.isfile(pose_txt):
                    continue

                with open(pose_txt, "r") as f:
                    pose_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

                reg_dir = osp.join(cam_dir, "registered") if self.USE_REGISTERED else cam_dir
                color_files = sorted(glob(osp.join(reg_dir, "*_color.png")))
                for cf in color_files:
                    bn = osp.basename(cf)
                    # 文件名形如 "3_color.png" → idx = 3
                    idx_str = bn.split("_")[0]
                    try:
                        idx = int(idx_str)
                    except Exception:
                        # 也可能是 "0003_color.png"；兼容
                        idx = int(idx_str.lstrip("0") or "0")

                    idx4 = f"{idx:04d}"
                    depth_path   = osp.join(reg_dir, f"{idx4}_depth.tiff")
                    normals_path = osp.join(reg_dir, f"{idx4}_normals.tiff")
                    occ_path     = osp.join(reg_dir, f"{idx4}_occlusion.png")

                    # 位姿：按 idx 对齐到对应行
                    if idx >= len(pose_lines):
                        continue
                    extri_w2c = parse_pose_row_major_4x4(pose_lines[idx], assume=self.assume_pose)

                    # 内参：如果没有 per-frame 提供，则用默认/FOV估计
                    # 先读图片分辨率以便估焦距
                    im0 = cv2.imread(cf)
                    if im0 is None:
                        continue
                    H, W = im0.shape[:2]

                    if (self.default_fx is not None) and (self.default_fy is not None):
                        fx, fy = float(self.default_fx), float(self.default_fy)
                    else:
                        fov = float(self.default_fov_deg) if (self.default_fov_deg is not None) else 60.0
                        # 用较大边估计（与很多实现一致）
                        fx = fy = (max(H, W) * 0.5) / np.tan(np.deg2rad(fov * 0.5))

                    cx = float(self.default_cx) if (self.default_cx is not None) else (W * 0.5)
                    cy = float(self.default_cy) if (self.default_cy is not None) else (H * 0.5)

                    K = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0,  0,  1]], dtype=np.float64)

                    seq_items.append(dict(
                        frame_idx=idx,
                        color_path=cf,
                        depth_path=depth_path,
                        normals_path=normals_path,
                        occ_path=occ_path,
                        extri_w2c=extri_w2c,
                        K=K,
                        H=H, W=W,
                        cam_name=cam,
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
        返回与 Co3dDataset.get_data 完全一致的 batch：
          {
            "seq_name", "ids", "frame_num",
            "images", "depths",
            "extrinsics", "intrinsics",
            "cam_points", "world_points", "point_masks",
            "original_sizes"
          }
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

            # 1) 读图
            image = read_image_cv2(color_path)
            if image is None:
                # 遇到坏图，简单跳过（也可换成 raise）
                continue

            # 2) 读深度（可选）并清理遮挡
            if self.load_depth and osp.isfile(anno["depth_path"]):
                depth_map = read_depth_any(anno["depth_path"], self.depth_unit_scale)
                # 用 occlusion 把无效像素清零（注意正负定义，必要时改阈值反转）
                if osp.isfile(anno["occ_path"]):
                    occ = cv2.imread(anno["occ_path"], cv2.IMREAD_GRAYSCALE)
                    if occ is not None:
                        # 这里假设 occ>0 为“遮挡/无效”
                        depth_map[occ > 0] = 0.0
                # 分位截断（去极端大/小值）
                depth_map = threshold_depth_map(depth_map, max_percentile=98, min_percentile=-1)
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])

            # 3) 位姿/内参（OpenCV 约定；外参为 w->c 3x4）
            extri_opencv = np.array(anno["extri_w2c"], dtype=np.float64)
            intri_opencv = np.array(anno["K"], dtype=np.float64)

            # 4) 交给基类的“一站式”预处理：
            #    - 随机尺度（训练态）
            #    - 以主点为中心裁剪
            #    - 缩放到 target_image_shape，并同步改写 K
            #    - 必要时 90° 旋转以适配纵横比
            #    - 基于 (depth, K, extri) 计算相机/世界坐标点云与有效掩膜
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

        batch = {
            "seq_name": "c3vd_" + seq_name,
            "ids": ids,
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
        return batch
