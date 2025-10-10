import torch
from typing import Tuple

def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32   # CPU 上用 float32
    return device, dtype


def load_vggt(device: str):
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model

def load_yaml

