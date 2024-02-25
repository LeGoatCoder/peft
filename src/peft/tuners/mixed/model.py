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

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import torch
from tqdm import tqdm

from peft.tuners import adalora, loha, lokr, lora, oft
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _get_submodules,
    get_auto_gptq_quant_linear,
)

# Collection of constants used for all tuners
COMPATIBLE_TUNER_TYPES = (PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA, PeftType.OFT)
PREFIXES = [lora.LoraModel.prefix, lokr.LoKrModel.prefix, loha.LoHaModel.prefix, oft.OFTModel.prefix]
Configs = Union[lora.LoraConfig, loha.LoHaConfig, lokr.LoKrConfig, adalora.AdaLoraConfig, oft.OFTConfig]
Layers = (lora.layer.LoraLayer, loha.layer.LoHaLayer, lokr.layer.LoKrLayer, adalora.layer.AdaLoraLayer, oft.OFTLayer)

class MixedModel(BaseTuner):
    """
    A class that allows to mix different types of adapters in a single model.

    Note: This class should usually not be initialized directly. Instead, use `get_peft_model` with the argument
    `mixed=True`.

    Args:
        model (nn.Module):
            The model to be tuned.
        config (PeftConfig):
            The config of the model to be tuned. The adapter type must be compatible.
        adapter_name (str):
            The name of the first adapter.
    """

    def __init__(self, model: nn.Module, config: Configs, adapter_name: str) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: Configs) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        if not isinstance(config, Configs.__args__):
            raise ValueError(
                f"{self.__class__.__name__} only supports {COMPATIBLE_TUNER_TYPES} configs, but got {type(config)}."
            )

        biases = (getattr(config, "bias", None) for config in self.peft_config)
        biases = [bias for bias in biases if bias not in (None, "none")]
        if len(biases) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(config: Configs, key: str):
        return check_target_module_exists(config, key)

    def _create_and_replace(
        self,
        config: Configs,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Creates a new adapter module and replaces the target module with the new adapter module.

        Args:
            config (Configs):
                The adapter config.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.

        """
        if isinstance(config, adalora.AdaLoraConfig):
            adalora.AdaLoraModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, lora.LoraConfig):
            lora.LoraModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, loha.LoHaConfig):
            loha.LoHaModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, lokr.LoKrConfig):
            lokr.LoKrModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, oft.OFTConfig):
            oft.OFTModel._create_and_replace(self, config, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")

    def _replace_module(self, parent, child_name, new_module, child) -> None:
        """
        Replaces the target module with the new adapter module.

        Args:
            parent (nn.Module):
                The parent module of the target module.
            child_name (str):
                The name of the target module.
            new_module (nn.Module):
                The new adapter module.
            child (nn.Module):
                The target module.


