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

import copy
import itertools
import os
import re
import tempfile
import unittest

import pytest
import torch
from parameterized import parameterized
from torch import nn
from transformers import AutoModelForCausalLM

# Import the PeFT modules
from peft import (
    AdaLoraConfig,  # Config for AdaLora tuner
    LoHaConfig,  # Config for LoHa tuner
    LoKrConfig,  # Config for LoKr tuner
    LoraConfig,  # Config for LoRa tuner
    OFTConfig,  # Config for OFT tuner
    PeftMixedModel,  # A mixed model that can have multiple tuners
    PrefixTuningConfig,  # Config for Prefix Tuning tuner
    get_peft_model,  # Function to get a PeFT model
)
from peft.tuners.tuners_utils import BaseTunerLayer  # Base class for tuner layers
from peft.utils import infer_device  # Function to infer the device


class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        # note: out_features must be > rank or else OFT will be an identity transform
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        return X


def _param_name_func(testcase_func, param_num, params):
    # for parameterized tests in TextMixedAdapterTypes
    config0, config1 = params[0]
    name0 = config0.__class__.__name__[: -len("Config")]
    name1 = config1.__class__.__name__[: -len("Config")]
    if name0 != name1:
        return f"{testcase_func.__name__}_{param_num}_{name0}_{name1}"
    return f"{testcase_func.__name__}_{param_num}_{name0}_x2"


class TestMixedAdapterTypes(unittest.TestCase):
    torch_device = infer_device()

    def _get_model(self, model_cls, peft_config=None, adapter_name=None, seed=0, mixed=True):
        """
        Initialize a model with a PeFT config.

        :param model_cls: The class of the base model
        :param peft_config: The PeFT config
        :param adapter_name: The name of the adapter
        :param seed: The random seed
        :param mixed: Whether to use a mixed model
        :return: The initialized model
        """
        torch.manual_seed(0)  # always use seed 0 for base model, seed for adapters may differ
        base_model = model_cls().eval().to(self.torch_device)
        if peft_config is None:
            return base_model

        torch.manual_seed(seed)
        assert adapter_name is not None
        peft_model = get_peft_model(base_model, peft_config, adapter_name=adapter_name, mixed=mixed)
        return peft_model.eval().to(self.torch_device)

    def _check_mixed_outputs(self, model_cls, config0, config1, input, *, is_commutative):
        """
        Check the outputs of mixed models with different adapter configurations.

        :param model_cls: The class of the base model
        :param config0: The first PeFT config
        :param config1: The second PeFT config
        :param input: The input to the model
        :param is_commutative: Whether the order of adapters should be commutative
        """
        atol = 1e-5
        rtol = 1e-5
        seed0 = 0
        seed1 = 1

        # base model
        base_model = self._get_model(model_cls)
        output_base = base_model(input)
        assert torch.isfinite(output_base).all()

        # adapter 0
        peft_model_0 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        output_config0 = peft_model_0(input)

        assert torch.isfinite(output_config0).all()
        assert not torch.allclose(output_base, output_config0, atol=atol, rtol=rtol)

        # adapter 1
        peft_model_1 = self._get_model(model_cls, config1, "adapter1", seed=seed1)
        output_config1 = peft_model_1(input)

        assert torch.isfinite(output_config1).all()
        assert not torch.allclose(output_base, output_config1, atol=atol, rtol=rtol)
        assert not torch.allclose(output_config0, output_config1, atol=atol, rtol=rtol)

        # adapter 0 + 1
        peft_model_01 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        torch.manual_seed(seed1)
        peft_model_0
