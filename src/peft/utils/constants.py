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

import torch

# This function is used to postprocess past key values for the Bloom model
# in the context of prefix-tuning.
def bloom_model_postprocess_past_key_value(past_key_values):
    # Concatenate the past key values along the first dimension
    past_key_values = torch.cat(past_key_values)
    
    # Get the shapes of the past key values
    total_layers, batch_size, num_attention_heads, num_virtual_tokens, head_dim = past_key_values.shape
    
    # Split the past key values into keys and values
    keys = past_key_values[:total_layers // 2]
    values = past_key_values[total_layers // 2:]
    
    # Reshape the keys and values to be compatible with the Bloom model
    keys = keys.transpose(2, 3).reshape(
        total_layers // 2, batch_size * num_attention_heads, head_dim, num_virtual_tokens
    )
    values = values.reshape(total_layers // 2, batch_size * num_attention_heads, num_virtual_tokens, head_dim)

    # Return the reshaped keys and values as a tuple
    return tuple(zip(keys, values))

# This function is used to postprocess past key values for the StarCoder models
# in the context of prefix-tuning.
def starcoder_model_postprocess_past_key_value(past_key_values):
    # Initialize an empty list to store the processed key values
    result = []
    
    # Iterate over each key value in the input
    for k in past_key_values:
        # Select the first token of the key value
        k = k[:, :, 0]
        
        # Permute the dimensions of the key value
        k = k.permute([1, 2, 0, 3])
        
        # Flatten the key value along the last two dimensions
        k = k.reshape(*k.shape[:-2], -1)
        
        # Add the processed key value to the list
        result.append(k)
    
    # Return the list of processed key values as a tuple
    return tuple(result)

# This dictionary maps Transformers model names to their corresponding
# prefix-tuning postprocessing functions.
TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {
    "bloom": bloom_model_postprocess_past_key_value,
    "gpt_bigcode": starcoder_model_postprocess_past_key_value,
}

# This dictionary maps Transformers model names to their corresponding
# LoRa target modules.
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj", "fc1", "fc2"],
    "gemma": ["q_proj", "v_proj"],
}

# This dictionary maps Transformers model names to their corresponding
# IA3 target modules.
TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {
    "t5": ["k", "v", "wo"],
    "mt5": ["k", "v", "wi_1"],
    "gpt2": ["c_attn", "mlp.c_proj"],
    "bloom": ["query_key_value", "mlp.dense_4h_to_h"],
    "roberta": ["key", "value", "output.dense"],
    "opt": ["
