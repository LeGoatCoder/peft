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
import os
import pickle
import tempfile
import unittest
import warnings

import pytest
from parameterized import parameterized

from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    IA3Config,
    LoHaConfig,
    LoraConfig,
    MultitaskPromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoder,
    PromptEncoderConfig,
    PromptTuningConfig,
    OFTConfig,
    PeftConfig,
    PeftType,
    PolyConfig,
)

# Define the models to test
PEFT_MODELS_TO_TEST = [("lewtun/tiny-random-OPTForCausalLM-delta", "v1")]

# Define all config classes to test
ALL_CONFIG_CLASSES = (
    AdaptionPromptConfig,
    AdaLoraConfig,
    IA3Config,
    LoHaConfig,
    LoraConfig,
    MultitaskPromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    OFTConfig,
    PolyConfig,
)

class PeftConfigTester(unittest.TestCase):
    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_methods(self, config_class):
        # Test if all configs have the expected methods
        config = config_class()
        assert hasattr(config, "to_dict")
        assert hasattr(config, "save_pretrained")
        assert hasattr(config, "from_pretrained")
        assert hasattr(config, "from_json_file")

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_task_type(self, config_class):
        # Test if all configs can be initialized with a task_type argument
        config_class(task_type="test")

    def test_from_peft_type(self):
        # Test if the config is correctly loaded using:
        # - from_peft_type
        from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        for peft_type in PeftType:
            expected_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
            config = PeftConfig.from_peft_type(peft_type=peft_type)
            assert type(config) is expected_cls

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_from_pretrained(self, config_class):
        # Test if the config is correctly loaded using:
        # - from_pretrained
        for model_name, revision in PEFT_MODELS_TO_TEST:
            # Test we can load config from delta
            config_class.from_pretrained(model_name, revision=revision)

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_save_pretrained(self, config_class):
        # Test if the config is correctly saved and loaded using
        # - save_pretrained
        config = config_class()
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_pretrained = config_class.from_pretrained(tmp_dirname)
            assert config.to_dict() == config_from_pretrained.to_dict()

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_from_json_file(self, config_class):
        config = config_class()
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_json = config_class.from_json_file(os.path.join(tmp_dirname, "adapter_config.json"))
            assert config.to_dict() == config_from_json

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_to_dict(self, config_class):
        # Test if the config can be correctly converted to a dict using:
        # - to_dict
        config = config_class()
        assert isinstance(config.to_dict(), dict)

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_from_pretrained_cache_dir(self, config_class):
        # Test if the config is correctly loaded with extra kwargs
        with tempfile.TemporaryDirectory() as tmp_dirname:
            for model_name, revision in PEFT_MODELS_TO_TEST:
                # Test we can load config from delta
                config_class.from_pretrained(model_name, revision=revision, cache_dir=tmp_dirname)

    def test_from_pretrained_cache_dir_remote(self):
        # Test if the config is correctly loaded with a checkpoint from the hub
        with tempfile.TemporaryDirectory() as tmp_dirname:
            PeftConfig.from_pretrained("ybelkada/test-st-lora", cache_dir=tmp_dirname)
            assert "models--ybelkada--test-st-lora" in os.listdir(tmp_dirname)

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_set_attributes(self, config_class):
        # Test if the config attributes are correctly set
        config = config_class(peft_type="test")

        #
