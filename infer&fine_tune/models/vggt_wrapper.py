# models/vggt_wrapper.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import torch


# -------------------------
# device / dtype 策略
# -------------------------
def pick_device(device_cfg: str) -> Tuple[str, torch.dtype]:
    """
    选择设备与AMP dtype策略：
      - device: "cuda" (若可用) 否则 "cpu"
      - amp_dtype: bfloat16 (>= SM80) 否则 float16；CPU上用 float32
    """
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float32
    return device, amp_dtype


# -------------------------
# VGGT 封装（推理）
# -------------------------
class VGGTWrapper:
    """
    轻量模型封装：
      - 负责 from_pretrained 加载
      - 提供 infer_images(images) 接口
      - 对模型输出做轻度标准化（统一 'depths' 键）
    """
    def __init__(self, device: str, amp_dtype: torch.dtype, amp_enabled: bool = True,
                 pretrained_id: str = "facebook/VGGT-1B"):
        from vggt.models.vggt import VGGT
        self.device = device
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

        # 允许后续通过 cfg 传入不同的权重id；当前骨干里直接使用默认值
        self.model = VGGT.from_pretrained(pretrained_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def infer_images(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,   # ← 新增：可选查询点
    ) -> Dict[str, Any]:
        """
        images: [S,3,H,W] 或 [B,S,3,H,W]，范围[0,1]
        query_points: [N,2] 或 [B,N,2]（像素坐标，可为 None）
        """
        # 直接透传到原始 VGGT 前向；VGGT 在有 query_points 时会返回 track/vis/conf
        return self.model(images, query_points=query_points)


# -------------------------
# 微调相关（占位实现）
#   * 推理模式不会调用
#   * 若误用会抛出 NotImplementedError
# -------------------------
def build_optimizer(model: VGGTWrapper, cfg: dict):
    """占位：后续实现优化器构建。"""
    raise NotImplementedError("build_optimizer is not implemented yet.")

def build_scheduler(optimizer, cfg: dict):
    """占位：后续实现学习率调度器。"""
    return None  # 可先返回 None，避免上层判空失败

def training_step(model: VGGTWrapper, batch, device: str, scaler_or_none, amp_dtype: torch.dtype, cfg: dict) -> Dict[str, float]:
    """占位：后续实现一次训练步（前向、反传、step）。"""
    raise NotImplementedError("training_step is not implemented yet.")

def eval_step(model: VGGTWrapper, batch, device: str, cfg: dict) -> Dict[str, float]:
    """占位：后续实现验证前向与指标计算。"""
    raise NotImplementedError("eval_step is not implemented yet.")

def save_checkpoint(model: VGGTWrapper, optimizer, scheduler, epoch: int, out_dir: Path) -> None:
    """占位：后续实现保存权重/优化器/调度器状态。"""
    raise NotImplementedError("save_checkpoint is not implemented yet.")
