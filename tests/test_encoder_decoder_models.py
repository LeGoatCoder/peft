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

import tempfile  # Tempfile module is used for creating temporary files and directories

import unittest
from parameterized import parameterized  # Parameterized module is used for parametrized testing

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

from peft import LoraConfig, TaskType, get_peft_model

from .testing_common import PeftCommonTester, PeftTestConfigManager

# Define the list of models to test
PEFT_ENCODER_DECODER_MODELS_TO_TEST = [
    "ybelkada/tiny-random-T5ForConditionalGeneration-calibrated",
    "hf-internal-testing/tiny-random-BartForConditionalGeneration",
]

# Define the full grid of parameters for testing
FULL_GRID = {
    "model_ids": PEFT_ENCODER_DECODER_MODELS_TO_TEST,
    "task_type": "SEQ_2_SEQ_LM"
}

class PeftEncoderDecoderModelTester(unittest.TestCase, PeftCommonTester):
    r"""
    Test if the PeftModel behaves as expected. This includes:
    - test if the model has the expected methods

    We use parametrized.expand for debugging purposes to test each model individually.
    """

    transformers_class = AutoModelForSeq2SeqLM

    def prepare_inputs_for_testing(self):
        # Prepare input tensors for testing
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        decoder_input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        input_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
        }

        return input_dict

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_attributes_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the model has the expected attributes
        self._test_model_attr(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_adapter_name(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the adapter has the expected name
        self._test_adapter_name(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_prepare_for_training_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the model prepares for training as expected
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the model saves pretrained as expected
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_pickle(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the model saves pretrained as a pickle file
        self._test_save_pretrained(model_id, config_cls, config_kwargs, safe_serialization=False)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_selected_adapters(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the model saves pretrained with selected adapters
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_selected_adapters_pickle(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the model saves pretrained with selected adapters as a pickle file
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs, safe_serialization=False)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_from_pretrained_config_construction(self, test_name, model_id, config_cls, config_kwargs):
        # Test if the model can be constructed from a pretrained config
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get
