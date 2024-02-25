import argparse
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

import safetensors
import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# Default kohya_ss LoRA replacement modules
# https://github.com/kohya-ss/sd-scripts/blob/c924c47f374ac1b6e33e71f82948eb1853e2243f/networks/lora.py#L661
UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"


@dataclass
class LoRAInfo:
    """
    A dataclass to store LoRA information for a specific module.
    """
    kohya_key: str
    peft_key: str
    alpha: Optional[float] = None
    rank: Optional[int] = None
    lora_A: Optional[torch.Tensor] = None
    lora_B: Optional[torch.Tensor] = None

    def peft_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Returns a state dictionary for the LoRA information that can be used with the PEEFT model.
        """
        if self.lora_A is None or self.lora_B is None:
            raise ValueError("At least one of lora_A or lora_B is None, they must both be provided")
        return {f"{peft_key}.lora_A.weight": self.lora_A, f"{peft_key}.lora_B.weight": self.lora_A}


def construct_peft_loraconfig(info: Dict[str, LoRAInfo]) -> LoraConfig:
    """
    Constructs LoraConfig from data extracted from kohya checkpoint

    Args:
        info (Dict[str, LoRAInfo]): Information extracted from kohya checkpoint

    Returns:
        LoraConfig: config for constructing LoRA
    """
    # Unpack all ranks and alphas
    ranks = {x[0]: x[1].rank for x in info.items()}
    alphas = {x[0]: x[1].alpha or x[1].rank for x in info.items()}

    # Determine which modules needs to be transformed
    target_modules = list(info.keys())

    # Determine most common rank and alpha
    r = Counter(ranks.values()).most_common(1)[0]
    lora_alpha = Counter(alphas.values()).most_common(1)[0]

    # Determine which modules have different rank and alpha
    rank_pattern = dict(filter(lambda x: x[1] != r, ranks.items()))
    alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, alphas.items()))

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=False,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    return config


def combine_peft_state_dict(info: Dict[str, LoRAInfo]) -> Dict[str, torch.Tensor]:
    """
    Combines the state dictionaries of LoRA information into a single state dictionary.

    Args:
        info (Dict[str, LoRAInfo]): Information extracted from kohya checkpoint

    Returns:
        Dict[str, torch.Tensor]: Combined state dictionary
    """
    result = {}
    for key_name, key_info in info.items():
        result[f"base_model.model.{key_name}.lora_A.weight"] = key_info.lora_A
        result[f"base_model.model.{key_name}.lora_B.weight"] = key_info.lora_B
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_checkpoint", default=None, type=str, required=True, help="SD checkpoint to use")

    parser.add_argument(
        "--kohya_lora_path", default=None, type=str, required=True, help="Path to kohya_ss trained LoRA"
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    # Load all models that we need to add adapter to
    text_encoder = CLIPTextModel.from_pretrained(args.sd_checkpoint, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.sd_checkpoint, subfolder="unet")

    # Construct possible mapping from kohya keys to peft keys
    models_keys = {}
    for model, model_key, model_name in [
        (text_encoder, LORA_PREFIX_TEXT_ENCODER, "text_encoder"),
        (unet, LORA_PREFIX_UNET, "unet"),
    ]:
        models_
