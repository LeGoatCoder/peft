# This is a docstring that describes what the code does. It is written in reStructuredText format.
# It provides information about the code's purpose, its inputs and outputs, and any assumptions or limitations.

"""
Copyright 2023-present the HuggingFace Inc. team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
import os
import tempfile
import unittest

import pytest
import torch
from parameterized import parameterized
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft import AdaLoraConfig, IA3Config, LoHaConfig, LoKrConfig, LoraConfig, OFTConfig, PeftModel, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper

# This is a module-level docstring that describes the contents of the module.
# It provides information about the module's purpose, its functions and classes, and any dependencies or requirements.
"""
This module contains unit tests for the `peft` module.

It includes tests for various aspects of the module, such as the correctness of the model's forward pass,
the saving and loading of models, the merging and unloading of adapters, and the behavior of multiple active adapters.

The tests are implemented using the `unittest` and `pytest` frameworks, and they use the `parameterized` decorator
to run the same test with different parameters.

The tests cover the following classes and functions:
- `PeftModel`: a class that represents a model with one or more adapters.
- `get_peft_model`: a function that creates a `PeftModel` instance from a base model and a configuration.
- `BaseTunerLayer`: a base class for adapter layers.
- `ModulesToSaveWrapper`: a class that wraps a module and its parameters and exposes them as separate objects.

The tests also cover the following configurations:
- `LoraConfig`: a configuration for LoRA adapters.
- `IA3Config`: a configuration for IA³ adapters.
- `LoHaConfig`: a configuration for LoHa adapters.
- `LoKrConfig`: a configuration for LoKr adapters.
- `OFTConfig`: a configuration for OFT adapters.

The tests use the following models:
- `MLP`: a simple multi-layer perceptron.
- `Block`: a building block for deep MLPs.
- `DeepMLP`: a deep multi-layer perceptron.
- `ModelEmbConv1D`: a model with an embedding layer and a 1D convolution layer.
- `ModelEmbWithEmbeddingUtils`: a model with an embedding layer and a `get_input_embeddings` method.
- `ModelConv2D`: a model with a 2D convolution layer.

The tests use the following classes and functions from the `transformers` library:
- `MockTransformerWrapper`: a mock class that behaves like a transformers model.
- `PeftCustomModelTester`: a base class for custom model testers.
"""

