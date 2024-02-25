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

from dataclasses import dataclass, field
from typing import List, Optional, Union

# Importing necessary classes and functions from peft package
from peft.tuners.lycoris_utils import LycorisConfig
from peft.utils import PeftType

@dataclass
class LoHaConfig(LycorisConfig):
    """
    This is the configuration class to store the configuration of a [`LoHaModel`].

    Args:
        r (int):
            LoHa rank.
        alpha (int):
            The alpha parameter for LoHa scaling.
        rank_dropout (float):
            The dropout probability for rank dimension during training.
        module_dropout (float):
            The dropout probability for disabling LoHa modules during training.
        use_effective_conv2d (bool):
            Use parameter effective decomposition for Conv2d with ksize > 1 ("Proposition 3" from FedPara paper).
        target_modules (Optional[Union[List[str], str]]):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        init_weights (bool):
            Whether to perform initialization of adapter weights. This defaults to `True`, passing `False` is
            discouraged.
        layers_to_transform (Union[List[int], int]):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (str):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        rank_pattern (dict):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (dict):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `alpha`.
        modules_to_save (Optional[List[str]]):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    # LoHa rank
    r: int = field(default=8, metadata={"help": "LoHa rank"})
    # The alpha parameter for LoHa scaling
    alpha: int = field(default=8, metadata={"help": "LoHa alpha"})
    # The dropout probability for rank dimension during training
    rank_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for rank dimension during training"}
    )
    # The dropout probability for disabling LoHa modules during training
    module_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for disabling LoHa modules during training"}
    )
    # Use parameter effective decomposition for Conv2d with ksize > 1 ("Proposition 3" from FedPara paper)
