#!/usr/bin/env python3

# coding=utf-8
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

import os # Importing os module to interact with operating system
import tempfile # Importing tempfile module to create temporary files and directories
import unittest # Importing unittest module to create test cases

import torch # Importing torch module for deep learning
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # Importing necessary classes from transformers library

from peft import PeftModel, PolyConfig, TaskType, get_peft_model # Importing necessary classes from peft library

class TestPoly(unittest.TestCase): # Defining a test case for the poly method
    def test_poly(self): # Defining a test method
        torch.manual_seed(0) # Setting seed for reproducibility
        model_name_or_path = "google/flan-t5-small" # Defining the name or path of the pretrained model

        # Defining the absolute and relative tolerances for the test
        atol, rtol = 1e-6, 1e-6

        # Defining the hyperparameters for the poly method
        r = 8  # rank of lora in poly
        n_tasks = 3  # number of tasks
        n_skills = 2  # number of skills (loras)
        n_splits = 4  # number of heads
        lr = 1e-2 # Learning rate
        num_epochs = 10 # Number of epochs for training

        # Initializing the tokenizer for the pretrained model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Initializing the base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        # Initializing the configuration for the poly method
        peft_config = PolyConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            poly_type="poly",
            r=r,
            n_tasks=n_tasks,
            n_skills=n_skills,
            n_splits=n_splits,
        )

        # Initializing the poly model
        model = get_peft_model(base_model, peft_config)

        # Generating some dummy data for training
        text = os.__doc__.splitlines()
        assert len(text) > 10
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs["task_ids"] = torch.arange(len(text)) % n_tasks
        inputs["labels"] = tokenizer((["A", "B"] * 100)[: len(text)], return_tensors="pt")["input_ids"]

        # Training the poly model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        for _ in range(num_epochs):
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Checking if the loss has improved by at least 50%
        assert losses[-1] < (0.5 * losses[0])

        # Checking if saving and loading the model works
        torch.manual_seed(0)
        model.eval()
        logits_before = model(**inputs).logits
        tokens_before = model.generate(**inputs)

        # Checking if disabling the adapter works
        with model.disable_adapter():
            logits_disabled = model(**inputs).logits
            tokens_disabled = model.generate(**inputs)

        # Checking if the logits and tokens are different when the adapter is disabled
        assert not torch.allclose(logits_before, logits_disabled, atol=atol, rtol=rtol)
        assert not torch.allclose(tokens_before, tokens_dis
