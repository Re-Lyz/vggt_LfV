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
    def __init__(
        self,
        device: str,
        amp_dtype: torch.dtype,
        amp_enabled: bool = True,
        pretrained_id: str = "facebook/VGGT-1B",
        resume: Optional[str] = None,          
        strict: bool = True,                    
    ):
        from vggt.models.vggt import VGGT
        self.device = device
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

        # 1) 先构建模型（用官方预训练权重或者随机初始化骨干）
        #    - 如果你想完全随机结构，可改成 VGGT(enable_xxx=...) 再手动 load
        self.model = VGGT.from_pretrained(pretrained_id).to(device)

        # 2) 如配置了 resume，则覆盖加载自定义权重
        if resume:
            self._load_checkpoint(Path(resume), strict=strict)

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
    
    def _load_checkpoint(self, ckpt_path: Path, strict: bool = True) -> None:
        """
        支持：
          - 直接文件：.pth/.pt（torch.save state_dict 或 {"state_dict": ...}）
          - safetensors：.safetensors
          - 目录：包含 config.json + 模型权重（交给 from_pretrained 处理）
        自动清洗常见前缀：'state_dict' 包裹、'module.'、'model.'、'ema' 分支等。
        """
        if ckpt_path.is_dir():
            # 允许用 from_pretrained 加载本地目录（含权重和 config）
            try:
                from vggt.models.vggt import VGGT
                m = VGGT.from_pretrained(str(ckpt_path), local_files_only=True)
                self.model.load_state_dict(m.state_dict(), strict=strict)
                del m
                print(f"[VGGTWrapper] Loaded weights from directory: {ckpt_path}")
                return
            except Exception as e:
                print(f"[VGGTWrapper] Failed directory from_pretrained: {e}; fallback to file scan.")
                # 目录兜底：找一个常见文件名
                for name in ("pytorch_model.bin", "model.safetensors", "weights.safetensors", "model.pth", "checkpoint.pth"):
                    p = ckpt_path / name
                    if p.is_file():
                        ckpt_path = p
                        break

        if ckpt_path.suffix.lower() in {".safetensors"}:
            try:
                from safetensors.torch import load_file as safe_load
                state = safe_load(str(ckpt_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load safetensors: {ckpt_path}") from e
        else:
            state = torch.load(str(ckpt_path), map_location="cpu")

        # 兼容各种包装
        if isinstance(state, dict):
            # 1) 典型：{"state_dict": {...}}
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            # 2) 可能有 EMA/best/… 分支，择优挑选
            for k in ("ema_state_dict", "model_ema", "module_ema"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break

        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint does not look like a state_dict: {ckpt_path}")

        # 清洗常见 key 前缀
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            cleaned[nk] = v

        # 加载
        missing, unexpected = self.model.load_state_dict(cleaned, strict=strict)
        # PyTorch 2.4+: load_state_dict 返回 NamedTuple；老版本返回 None
        if missing or unexpected:
            print(f"[VGGTWrapper] load_state_dict(strict={strict}) "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) < 20 and len(unexpected) < 20:
                print("  missing:", missing)
                print("  unexpected:", unexpected)
        print(f"[VGGTWrapper] Loaded custom checkpoint: {ckpt_path}")

