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

import importlib
import os
import tempfile
import unittest
from unittest import TestCase

import pytest
import torch
from torch.testing import assert_close

from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.adaption_prompt import AdaptionPromptConfig
from peft.utils.other import prepare_model_for_int8_training
from peft.utils.save_and_load import get_peft_model_state_dict
from tests.testing_common import PeftCommonTester


