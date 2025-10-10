#!/usr/bin/env python3
import argparse
import numpy as np
import pprint
from omegaconf import OmegaConf
import torch
import open3d as o3d  

# —— 你的工程内的导入（按实际包路径调整）
from data.datasets.c3vd import C3VDDatasetv1
# 如果 BaseDataset 的工具函数在 data.dataset_util 内部使用，这里不需要重复导入

def save_ply(points, colors, filename):
              
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

# Usage example


def main():
    parser = argparse.ArgumentParser("C3VD quick check")
    parser.add_argument("--cfg", type=str, default="./config/custom_dataset.yaml", help="path to default.yaml")
    parser.add_argument("--root", type=str, default="/home/re_lyz/python/vggt_data/C3VD/registered", help="path to C3VD root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=3, help="images per sequence to fetch")
    parser.add_argument("--seq_index", type=int, default=None, help="optional fixed seq index")
    args = parser.parse_args()

    # 1) 读配置
    cfg = OmegaConf.load(args.cfg)
    
    # 取 train/val 的 common_config（两边你都已经写了 ocam 与开关）
    common_conf = cfg.data.train.common_config if args.split == "train" else cfg.data.val.common_config

    # 可选：为了多打印点信息，强制 debug=True
    common_conf.debug = True
    # 如果你想在验证时一定加载深度：
    common_conf.load_depth = True

    # 2) 实例化数据集
    ds = C3VDDatasetv1(
        common_conf=common_conf,
        split=args.split,
        ROOT=args.root,
        USE_REGISTERED=True,      # 若你的目录没有 registered 子目录，改成 False
        assume_pose="w2c",        # 若确认是 c2w，这里改成 "c2w"
        depth_unit_scale=1.0,     # 若深度是毫米→米，改成 0.001
    )

    print(f"[INFO] sequences found: {ds.sequence_list_len}")
    if ds.sequence_list_len == 0:
        raise RuntimeError("No valid sequences scanned. Check ROOT layout and pose.txt existence.")

    # 3) 取一个 batch
    batch = ds.get_data(
        seq_index=args.seq_index,
        img_per_seq=args.n,
        aspect_ratio=1.0,
    )
    print(batch)
    # save_ply(
    #     batch["world_points"][0].reshape(-1, 3), 
    #     torch.from_numpy(batch["images"][0]).permute(0, 2, 3, 1).reshape(-1, 3), 
    #     "debug.ply"
    # )

    # 4) 打印关键信息
    print("\n=== Batch summary ===")
    keys = [
        "seq_name", "ids", "frame_num",
        "images", "depths", "extrinsics", "intrinsics",
        "cam_points", "world_points", "point_masks", "original_sizes",
    ]
    for k in keys:
        if k in batch:
            if isinstance(batch[k], list):
                print(f"{k:>16s}: list(len={len(batch[k])})")
            else:
                print(f"{k:>16s}: {type(batch[k])}")
    print()

    # 5) 逐帧详细检查
    n = batch["frame_num"]
    for i in range(n):
        img = batch["images"][i]
        dep = batch["depths"][i]
        K   = batch["intrinsics"][i]
        Ext = batch["extrinsics"][i]
        msk = batch["point_masks"][i]
        H0, W0 = batch["original_sizes"][i]

        print(f"[Frame {i}] image={img.shape}, "
              f"depth={'None' if dep is None else dep.shape}, "
              f"H0xW0=({H0},{W0})")
        print("          K=\n", np.array_str(K, precision=3, suppress_small=True))
        print("          Ext shape:", Ext.shape)

        # 一些快速 sanity check
        assert K.shape == (3,3), "K must be 3x3"
        assert Ext.shape == (3,4), "Extrinsic must be 3x4"
        assert img.ndim == 3 and img.shape[2] in (3,4), "image must be HxWx3/4"

        if dep is not None:
            valid = (msk > 0).sum()
            total = msk.size
            ratio = valid / max(1, total)
            print(f"          valid points: {valid}/{total} ({ratio:.3f})")

    # 6) 可视化提示（可选）
    # 这里只做文字验证；如需可视化，可以把某帧 image/depth 保存出来对拍：
    # import cv2
    # cv2.imwrite("debug_img.png", cv2.cvtColor(batch["images"][0], cv2.COLOR_RGB2BGR))

    print("\n[OK] C3VDDatasetv1 quick check finished without errors.")

if __name__ == "__main__":
    main()
