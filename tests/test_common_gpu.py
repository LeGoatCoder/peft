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

import gc
import tempfile
import unittest

import pytest
import torch
import torch.nn.functional as F
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    WhisperForConditionalGeneration,
)

from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    IA3Config,
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.import_utils import is_bnb_4bit_available, is_bnb_available

# Check if BitsAndBytes library is available
if is_bnb_available():
    import bitsandbytes as bnb

    # Import Linear8bitLt and Linear4bit for IA3 and LoRa tuners
    from peft.tuners.ia3 import Linear8bitLt as IA3Linear8bitLt, Linear4bit as IA3Linear4bit
    from peft.tuners.lora import Linear8bitLt as LoraLinear8bitLt, Linear4bit as LoraLinear4bit

@require_torch_gpu  # Decorator to run the test only on GPU
class PeftGPUCommonTests(unittest.TestCase):
    r"""
    A common tester to run common operations that are performed on GPU such as generation, loading in 8bit, etc.
    """

    def setUp(self):
        r"""
        Set up the test environment.
        """
        self.seq2seq_model_id = "google/flan-t5-base"
        self.causal_lm_model_id = "facebook/opt-350m"
        self.audio_model_id = "openai/whisper-large"

        # Initialize the device to cuda:0 if GPU is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

    def tearDown(self):
        r"""
        Clean up the test environment by freeing GPU memory.
        """
        gc.collect()  # Collect the garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Empty the cache
        gc.collect()  # Collect the garbage again

    @require_bitsandbytes  # Decorator to run the test only if BitsAndBytes is available
    @pytest.mark.multi_gpu_tests  # Mark the test to run on multiple GPUs
    @pytest.mark.single_gpu_tests  # Mark the test to run on a single GPU
    def test_lora_bnb_8bit_quantization(self):
        r"""
        Test that tests if the 8bit quantization using LoRA works as expected
        """
        whisper_8bit = WhisperForConditionalGeneration.from_pretrained(
            self.audio_model_id,
            device_map="auto",
            load_in_8bit=True,
        )

        opt_8bit = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            device_map="auto",
            load_in_8bit=True,
        )

        flan_8bit = AutoModelForSeq2SeqLM.from_pretrained(
            self.seq2seq_model_id,
            device_map="auto",
            load_in_8bit=True,
        )

        # Initialize LoRaConfig for Seq2Seq model
        flan_lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
        )

        # Initialize LoRaConfig for CausalLM model
        opt_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Initialize LoRaConfig for general use
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

        # Apply LoRa tuning to models
        flan_8bit = get_peft_model(flan_8bit, flan_lora_config)
        assert isinstance(flan_8bit.base_model.model.encoder.block[0].layer[0].SelfAttention.q, LoraLinear8bitLt)

        opt_8bit = get_peft_model(opt_8bit, opt_lora_config)
        assert isinstance(opt_8bit.base_model.model.model.decoder.layers[0].self_attn.v_proj, LoraLinear8bitLt)

        whisper_8bit = get_peft_model(whisper_8bit, config)
