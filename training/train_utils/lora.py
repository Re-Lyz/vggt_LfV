# train_utils/lora.py
import logging
import re
from typing import Dict, Any, List, Optional

import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

def _expand_targets(blocks: Dict[str, List[int]], submods: List[str]) -> List[str]:
    """
    由 blocks/submodules 生成 PEFT target_modules 用的“包含匹配子串”列表。
    模块名示例：
      aggregator.global_blocks.23.attn.qkv
      aggregator.frame_blocks.21.attn.proj
    传入 target_modules 时使用“包含匹配”，因此给出上述子串即可精准命中。
    """
    targets = []
    for scope in ("frame_blocks", "global_blocks"):
        idxs = blocks.get(scope, []) or []
        for i in idxs:
            for s in submods:
                # 两种写法都可命中（保险起见都给）
                targets.append(f"{scope}.{i}.{s}")
                targets.append(f"aggregator.{scope}.{i}.{s}")
    # 去重
    return sorted(set(targets))


def _regex_for_full_train(blocks: Dict[str, List[int]], submods: List[str]) -> List[str]:
    """
    生成 re.search 用的白名单正则，精确到 attn.qkv / attn.proj 参数前缀。
    例：^aggregator\.global_blocks\.(4|11|17|23)\.attn\.(qkv|proj)\.
    """
    # submods 是 ["attn.qkv", "attn.proj"] 形式
    attn_names = [s.split("attn.")[1] if "attn." in s else s for s in submods]  # -> ["qkv","proj"]
    sub = "(" + "|".join(map(re.escape, attn_names)) + ")"
    patts = []
    for scope in ("frame_blocks", "global_blocks"):
        idxs = blocks.get(scope, [])
        if not idxs:
            continue
        idx_alt = "(" + "|".join(str(i) for i in idxs) + ")"
        patts.append(rf"^aggregator\.{scope}\.{idx_alt}\.attn\.{sub}\.")
    return patts


def _set_requires_grad_by_regex(model: nn.Module, regex_list: List[str], value: bool):
    for n, p in model.named_parameters():
        if any(re.search(pat, n) for pat in regex_list):
            p.requires_grad = value


def _force_train_extra_by_regex(model: nn.Module, regex_list: List[str]):
    if not regex_list:
        return
    for n, p in model.named_parameters():
        if any(re.search(pat, n) for pat in regex_list):
            p.requires_grad = True


def _log_trainable_stats(model: nn.Module, tag: str = "finetune"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"[{tag}] trainable = {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
    shown = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(f"[{tag}] + {n} {tuple(p.shape)}")
            shown += 1
            if shown >= 30:
                logging.info(f"[{tag}] ...")
                break


def apply_finetune_strategy(
    model: nn.Module,
    finetune_cfg: Optional[Dict[str, Any]],
) -> nn.Module:
    if not finetune_cfg or not finetune_cfg.get("enable", False):
        return model

    mode = str(finetune_cfg.get("mode", "lora")).lower()
    blocks = finetune_cfg.get("blocks", {}) or {}
    submods = finetune_cfg.get("submodules", ["attn.qkv", "attn.proj"])
    extra_regex = finetune_cfg.get("extra_trainable_regex", []) or []

    if mode == "lora":
        if not _PEFT_AVAILABLE:
            raise RuntimeError("LoRA mode requested but `peft` is not installed. `pip install -U peft`")

        targets = _expand_targets(blocks, submods)
        peft_cfg = finetune_cfg.get("peft", {}) or {}
        r = int(peft_cfg.get("r", 16))
        alpha = int(peft_cfg.get("alpha", r))
        dropout = float(peft_cfg.get("dropout", 0.0))
        bias = str(peft_cfg.get("bias", "none"))

        lora_conf = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,   # 关键：通用模型
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=targets,
            bias=bias,
        )
        peft_model = get_peft_model(model, lora_conf)

        # LoRA 模式下，PEFT 会把非 LoRA 参数全设为 requires_grad=False
        # 白名单：让头等模块全量参与训练
        _force_train_extra_by_regex(peft_model, extra_regex)
        

        logging.info(f"[finetune] PEFT-LoRA targets={targets} r={r} alpha={alpha} drop={dropout} bias={bias}")
        _log_trainable_stats(peft_model)
        return peft_model

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = False
        allow_attn = _regex_for_full_train(blocks, submods)
        _set_requires_grad_by_regex(model, allow_attn, True)
        _force_train_extra_by_regex(model, extra_regex)

        logging.info(f"[finetune] FULL allow_attn={allow_attn} extra={extra_regex}")
        _log_trainable_stats(model)
        return model

    logging.warning(f"[finetune] Unknown mode={mode}, skip finetune strategy.")
    return model


