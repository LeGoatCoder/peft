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

import tempfile  # Tempfile module is used to create temporary directories for saving models
import unittest  # Unit test framework for running test cases

import torch  # PyTorch library for tensor computations

from peft import (  # Importing various classes and functions from the peft module
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForFeatureExtraction,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    infer_device,
)
from peft.utils import infer_device  # Utility function to infer the device for torch tensors


class PeftAutoModelTester(unittest.TestCase):  # Subclass of unittest.TestCase for writing test cases
    dtype = torch.float16 if infer_device() == "mps" else torch.bfloat16  # Setting the data type based on the device

    def test_peft_causal_lm(self):  # Test function for AutoPeftModelForCausalLM
        model_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"  # Model ID for the pretrained model
        model = AutoPeftModelForCausalLM.from_pretrained(model_id)  # Loading the pretrained model
        assert isinstance(model, PeftModelForCausalLM)  # Checking if the model is an instance of PeftModelForCausalLM

        with tempfile.TemporaryDirectory() as tmp_dirname:  # Creating a temporary directory
            model.save_pretrained(tmp_dirname)  # Saving the model to the temporary directory

            model = AutoPeftModelForCausalLM.from_pretrained(tmp_dirname)  # Loading the model from the temporary directory
            assert isinstance(model, PeftModelForCausalLM)  # Checking if the model is an instance of PeftModelForCausalLM

        # Checking if kwargs are passed correctly
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype)
        assert isinstance(model, PeftModelForCausalLM)
        assert model.base_model.lm_head.weight.dtype == self.dtype

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForCausalLM.from_pretrained(model_id, adapter_name, is_trainable, torch_dtype=self.dtype)

    # Test functions for other AutoPeftModel variants follow a similar structure

if __name__ == "__main__":
    unittest.main()  # Running the test cases
