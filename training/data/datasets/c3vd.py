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
        len_test: int = 100,

        # 位姿方向：pose.txt 若是 c2w，则 assume_pose='c2w'；w2c 则 'w2c'
        assume_pose: str = "c2w",

        # 深度单位（如毫米→米）
        depth_unit_scale: float = 100.0/65535.0,
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
            for i_order, cf in enumerate(color_files):
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

                if not osp.isfile(cf):
                    if self.debug:
                        print(f"[Skip] color file not found: {cf}")
                    continue    

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
        # print("meta length:", len(meta))
        if ids is None:
            ids = np.random.choice(len(meta), img_per_seq, replace=self.allow_duplicate_img)

            # N = len(meta)
            # K = img_per_seq
            # edges = np.linspace(0, N, num=K+1, dtype=int)
            # ids = np.array([
            #     np.random.randint(edges[i], max(edges[i] + 1, edges[i + 1]))
            #     for i in range(K)
            # ])

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            anno = meta[int(i)]
            color_path = anno["color_path"]

            # 1) 读图（图像本身仍按实际尺寸加载；K 已按 common_conf 的 H,W 生成）
            image = read_image_cv2(color_path, rgb=False)
            if image is None:
                raise RuntimeError(f"Failed to read image: {color_path}")
            
            # save_image_and_depth(image, None, f"original_image_{i}.png", f"original_depth_{i}.png")
            
            # 2) 读深度（可选）并处理 occlusion
            if self.load_depth and osp.isfile(anno["depth_path"]):
                depth_map = read_depth_any(anno["depth_path"], self.depth_unit_scale)
                if osp.isfile(anno["occ_path"]):
                    occ = cv2.imread(anno["occ_path"], cv2.IMREAD_GRAYSCALE)
                    if occ is not None:
                        depth_map[occ > 0] = 0.0
                depth_map = threshold_depth_map(depth_map, max_percentile=98, min_percentile=-1)
                
                
                # save_image_and_depth(None, depth_map, f"original_image_{i}.png", f"original_depth_{i}.png")
            else:
                depth_map = None
                

            # import sys
            # np.set_printoptions(threshold=sys.maxsize, precision=4)
            # print("depth_map stats:", depth_map)
                
            original_size = np.array(image.shape[:2])

            # 3) 位姿/内参（OpenCV 约定；外参为 w->c 3x4）
            extri_opencv = np.array(anno["extri_w2c"], dtype=np.float64)
            intri_opencv = np.array(anno["K"], dtype=np.float64)
            # print("shape of image before process_one_image:", image.shape)
            # 4) 统一预处理与点云
            # check_camera_parameters(extri_opencv, intri_opencv)

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
            # check_camera_parameters(extri_opencv, intri_opencv)
            # save_image_and_depth(image, depth_map, f"processed_image_{i}.png", f"processed_depth_{i}.png")
            
            # # === 关键：对所有会被 collate 的返回做 C 连续与 dtype 统一 ===
            # image = np.ascontiguousarray(image, dtype=np.uint8)  # 或 float32，视你的下游而定
            # if depth_map is None:
            #     # 避免 None 参与 collate；按你的设计返回空阵或全 0，形状要一致
            #     depth_map = np.zeros(target_image_shape, dtype=np.float32)
            # else:
            #     depth_map = np.ascontiguousarray(depth_map, dtype=np.float32)

            # extri_opencv = np.ascontiguousarray(extri_opencv, dtype=np.float32)
            # intri_opencv = np.ascontiguousarray(intri_opencv, dtype=np.float32)

            # # 点云/掩膜也要保证连续 & dtype 统一（float32 / uint8）
            # if world_coords_points is not None:
            #     world_coords_points = np.ascontiguousarray(world_coords_points, dtype=np.float32)
            # if cam_coords_points is not None:
            #     cam_coords_points = np.ascontiguousarray(cam_coords_points, dtype=np.float32)
            # if point_mask is not None:
            #     point_mask = np.ascontiguousarray(point_mask, dtype=np.uint8)
            # # print("shape of image after process_one_image:", image.shape)
            
            # print_array_info(depth_map, "depth_map")
            # print_array_info(extri_opencv, "extri_opencv")
            # print_array_info(intri_opencv, "intri_opencv")

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
        # print("shape of batch['images']:", np.array(batch["images"]).shape)
        if getattr(self, "debug", False):
            batch["_debug_image_paths"] = [meta[int(i)]["color_path"] for i in ids]
            batch["_debug_depth_paths"] = [
                meta[int(i)]["depth_path"] if osp.isfile(meta[int(i)]["depth_path"]) else None
                for i in ids
            ]
            batch["_debug_occ_paths"] = [
                meta[int(i)]["occ_path"] if osp.isfile(meta[int(i)]["occ_path"]) else None
                for i in ids
            ]
        # print(f"[C3VD] got_frames={len(images)} target={img_per_seq}")
        return batch



def save_image_and_depth(image, depth_map, image_name, depth_name):
    """
    将图像和深度图保存到当前目录中。

    Args:
        image: 图像数据，通常是 BGR 或 RGB 格式。
        depth_map: 深度图数据。
        image_name: 保存图像的文件名。
        depth_name: 保存深度图的文件名。
    """
    # 保存图像
    path = os.getcwd()
    save_path = os.path.join(path, "debugging_outputs")
    os.makedirs(save_path, exist_ok=True)
    if image is not None:
        image_path = os.path.join(save_path, image_name)
        cv2.imwrite(image_path, image)
        print(f"Saved image to {image_path}")

    # 保存深度图
    if depth_map is not None:
        depth_path = os.path.join(save_path, depth_name)
        # 深度图通常是浮点类型，将其转换为适当的显示格式
        depth_map = np.uint16(depth_map * 65535 / 100)  # 将深度图范围映射到 16 位
        cv2.imwrite(depth_path, depth_map)
        print(f"Saved depth map to {depth_path}")

def check_camera_parameters(extri_opencv, intri_opencv):
    # 检查内参尺寸
    if intri_opencv.shape != (3, 3):
        print("Warning: intri_opencv should be a 3x3 matrix.")
        return False

    # 检查外参尺寸
    if extri_opencv.shape != (3, 4):
        print("Warning: extri_opencv should be a 3x4 matrix.")
        return False
    
    # 检查数据类型
    if not (extri_opencv.dtype == np.float64 or extri_opencv.dtype == np.float32):
        print("Warning: extri_opencv should be of type np.float64 or np.float32.")
        return False
    if not (intri_opencv.dtype == np.float64 or intri_opencv.dtype == np.float32):
        print("Warning: intri_opencv should be of type np.float64 or np.float32.")
        return False

    # 检查是否包含 NaN 或 Inf
    if np.any(np.isnan(extri_opencv)) or np.any(np.isinf(extri_opencv)):
        print("Warning: extri_opencv contains NaN or Inf values.")
        return False
    if np.any(np.isnan(intri_opencv)) or np.any(np.isinf(intri_opencv)):
        print("Warning: intri_opencv contains NaN or Inf values.")
        return False

    # 检查内参矩阵的合理性（例如焦距应为正）
    if not np.all(intri_opencv >= 0):
        print("Warning: intri_opencv contains non-positive values.")
        return False

    # 检查外参的旋转矩阵部分是否正交
    rotation_matrix = extri_opencv[:, :3]  # 提取旋转矩阵
    if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-6):
        print("Warning: Rotation matrix in extri_opencv is not orthogonal.")
        return False

    print("Camera parameters are valid.")
    return True

def print_array_info(arr, name="Array"):
    print(f"{name} Info:")
    print(f"  - Shape: {arr.shape}")
    print(f"  - Mean: {np.mean(arr)}")
    print(f"  - Min: {np.min(arr)}")
    print(f"  - Max: {np.max(arr)}")
    print(f"  - Std: {np.std(arr)}")
    print(f"  - Range: ({np.min(arr)}, {np.max(arr)})\n")

    
    