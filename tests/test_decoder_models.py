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

import unittest
from unittest.mock import Mock, call, patch

import pytest
import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import AdaLoraConfig, PromptTuningConfig, PromptTuningInit, get_peft_model

from .testing_common import PeftCommonTester, PeftTestConfigManager


