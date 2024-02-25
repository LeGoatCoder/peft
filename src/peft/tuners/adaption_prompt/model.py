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

import torch.nn as nn
from typing import Dict, List

from peft.utils import _freeze_adapter, _get_submodules
from .config import AdaptionPromptConfig, prepare_config
from .layer import AdaptedAttention
from .utils import is_adaption_prompt_trainable


class AdaptionPromptModel(nn.Module):
    """
    Implements adaption prompts as described in https://arxiv.org/pdf/2303.16199.pdf.

    The top L attention modules are replaced with AdaptedAttention modules that wrap the original ones, but insert
    trainable prompts with gates (for zero init).
    """

    def __init__(self, model, configs: Dict, adapter_name: str):
        super().__init__()
        # Save the original model
        self.model = model
        # Store adapter configs by name
        self.peft_config: Dict[str, AdaptionPromptConfig] = {}
        # Store lists of the parents of the affected attention modules by adapter name
        self._parents: Dict[str, List[nn.Module]] = {}
        # Store lists of cached AdaptedAttention modules by name
        self._cached_adapters: Dict[str, List] = {}
        # The name of the currently active adapter
        self._active_adapter = None
        # Whether the adapter is enabled
        self._enabled = True
        # Set forward method to the original model's forward method
        self.forward = self.model.forward
        # Add the first adapter
        self.add_adapter(adapter_name, configs[adapter_name])
        # Freeze all parameters of the model except the adaption prompts
        self._mark_only_adaption_prompts_as_trainable(self.model)

    def add_adapter(self, adapter_name: str, config: AdaptionPromptConfig) -> None:
        """Add an adapter with the given name and config."""
        config = prepare_config(config, self.model)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")

        parents = []
        for name, _ in self.model.named_modules():
            if name.endswith(config.target_modules):
                par, _, _ = _get_submodules(self.model, name)
                parents.append(par)
        if len(parents) < config.adapter_layers:
            raise ValueError(
                f"Config specifies more adapter layers '{config.adapter_layers}'"
                f" than the model has '{len(parents)}'."
            )
        # Note that if the target modules are not in Sequential, ModuleList, or
        # some other PyTorch ordered container, the behavior is undefined as we
        # assume here that the order of the modules is the same as the order of
        # the transformer decoder layers.
        parents = parents[-config.adapter_layers :]
        self._parents[adapter_name] = parents

        # It is only None during initialization.
        # If it is disabled, we don't have to remove the modules.
        if self._active_adapter is not None and self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
        self._active_adapter = adapter_name
        self.peft_config[adapter_name] = config
        self._create_adapted_attentions(config, parents)
        if not self._enabled:
            self._remove_adapted_attentions(self._active_adapter)

        if config.inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        """Set the model to use the adapter with the given name."""
        if self._active_adapter == adapter_name:
            return
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' does not exist.")

        if self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
            self._set_adapted_attentions(adapter_name)

        self._active_adapter = adapter_name

    def enable_adapter_layers(self):
        """Enable adapter layers by swapping in cached AdaptedAttention modules."""
        self._enabled = True
        self._set_adapted_attentions(self._active_adapter)

    def disable_adapter_layers(self):
        """Disable adapter layers by swapping out AdaptedAttention modules."""
        self._enabled = False
        self._remove_adapted_attentions(self._active_adapter)

    def _create_adapted_attentions(self, config: AdaptionPromptConfig, parents: List[nn.Module]) -> None:
        """Wrap LlamaAttention modules with newly created AdaptedAttention modules."""
        for par in parents:
            attn = AdaptedAttention(
                model_type=self.model.config.model_type,
                adapter_len=config.adapter_len,
                model=getattr(par, config.target_modules),
            )
            setattr(par, config.target_modules, attn)

    def _set_adapted_attentions(self, adapter_name: str) -> None:
        """Replace LlamaAttention modules with cached AdaptedAttention modules."""
