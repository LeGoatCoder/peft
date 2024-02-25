# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from typing import Optional

import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file

from .other import (
    EMBEDDING_LAYER_NAMES,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    check_file_exists_on_hf_hub,
    infer_device,
)
from .peft_types import PeftType

def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    return hasattr(layer, "base_layer") and isinstance(layer.base_layer, (torch.nn.Linear, torch.nn.Embedding))

def get_embedding_layer_name(model, layer, is_embedding_in_target_modules):
    """Get the name of the embedding module for a given layer."""
    for name, module in model.named_modules():
        if (not is_embedding_in_target_modules and module == layer) or module == getattr(layer, "base_layer", None):
            return name
    return None

def get_peft_model_state_dict(
    model, state_dict=None, adapter_name="default", unwrap_compiled=False, save_embedding_layers="auto"
):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    # The following if-elif block initializes the `to_return` dictionary
    # based on the type of the PEFT model.
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # The `to_return` dictionary is initialized with the LoRa or Adalora adapter weights
        # based on the `state_dict` of the model.
        pass
    elif config.peft_type == PeftType.LOHA:
        # The `to_return` dictionary is initialized with the LoHA adapter weights
        # based on the `state_dict` of the model.
        pass
    elif config.peft_type == PeftType.LOKR:
        # The `to_return` dictionary is initialized with the LOKR adapter weights
        # based on the `state_dict` of the model.
        pass
    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        # The `to_return` dictionary is initialized with the Adaption Prompt adapter weights
        # based on the `state_dict` of the model.
        pass
    elif config.is_prompt_learning:
        # The `to_return` dictionary is initialized with the Prompt Learning adapter weights
        # based on the `state_dict` of the model.
        pass
    elif config.peft_type == PeftType.IA3:
        # The `to_return` dictionary is initialized with the IA3 adapter weights
        # based on the `state_dict` of the model.
        pass
    elif config.peft_type == PeftType.OFT:
        # The `to_return` dictionary is initialized with the OFT adapter weights
        # based on the `state_dict` of the model.
        pass
    elif config.peft_type == PeftType.POLY:
        # The `to_return` dictionary is initialized with the POLY adapter weights
        # based on the `state_dict` of the model.
        pass
    else:
        raise NotImplementedError

    # The following if-else block checks if the `save_embedding_layers` parameter is set to `True`
    # or `"auto"` and if the embedding layers should be saved in the `to_return` dictionary.
    if save_embedding_layers:
        if save_embedding_layers == "auto":
            # The `save_embedding_layers` boolean flag is set based on the common embedding layers
            # in the `target_modules` of the config.
            pass
        if hasattr(model, "get_input_embeddings"):
            #
