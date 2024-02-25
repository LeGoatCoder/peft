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
import pickle
import re
import tempfile
import torch
import yaml
from dataclasses import replace

import pytest
from diffusers import StableDiffusionPipeline
from peft import (
    AdaLoraConfig,
    IA3Config,
    LoraConfig,
    PeftModel,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptLearningConfig,
    PromptTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from peft.tuners.lora import LoraLayer
from peft.utils import _get_submodules

from .testing_utils import get_state_dict


