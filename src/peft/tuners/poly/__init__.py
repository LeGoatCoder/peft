# Copyright 2023-present the HuggingFace Inc. team.
# 
# This code is licensed under the Apache License, Version 2.0 (the "License");
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

# Import the necessary modules and classes from the config, layer, and model files
from .config import PolyConfig
from .layer import Linear
from .model import PolyModel

# Define the list of all exported classes and functions from this module
__all__ = ["Linear", "PolyConfig", "PolyLayer", "PolyModel"]

# If the PolyLayer class needs to be defined, uncomment the following lines
# and implement the class

# from .layer import PolyLayer

# class PolyLayer(PolyLayer):
#     """
#     The PolyLayer class is a custom layer class that inherits from the base Layer class.
#     It implements the forward method to apply the polynomial transformation to the input.
#     """
#     def forward(self, input):
#         """
#         Apply the polynomial transformation to the input tensor.
#         :param input: the input tensor of shape (batch_size, sequence_length, hidden_size)
#         :return: the output tensor of shape (batch_size, sequence_length, hidden_size)
#         """
#         # Implement the polynomial transformation here
#         pass

