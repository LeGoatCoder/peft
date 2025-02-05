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

import inspect
import torch
import torch.nn as nn

def llama_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    This function was duplicated verbatim from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L126

    This was done to eliminate the Llama transformers implementation as a dependency of this file. Note that some other
    functions were also adapted from the transformers implementation but were modified.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def llama_apply_rotary_pos_emb(q, cos, sin, position_ids):
    """
    Apply rotary position embedding to query states in the Llama model.

    This function was adapted from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L133

    It was modified to remove unnecessary processing of key states. The method is compatible with transformers <=
    4.34.2 and also with the latest version (>=4.35).
    """
    # In previous transformers version cos/sin cached had a shape of 4D
    if len(cos.shape) == 4:
        gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    # In the new version, it is 2D so we fall back to the new implementation
    # https://github.com/huggingface/transformers/blame/eef7ea98c31a333bacdc7ae7a2372bde772be8e4/src/transformers/models/llama/modeling_llama.py#L222-L226
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (llama_rotate_half(q) * sin)
    return q_embed

def llama_compute_query_states(model: nn.Module, **kwargs) -> torch.Tensor:
    """
    Compute query states for Llama models specifically. They need to be recomputed as the forward() method of the
    original LlamaModel in the transformers library does not return them. See the related discussion in the PR:
    https://github.com/huggingface/peft/pull/268
    """
    hidden_states = kwargs.get("hidden_states")
    position_ids = kwargs.get("position_ids")
    past_key_value = kwargs.get("past_key_value")
    bsz, q_len, _ = hidden_states.size()

    # Compute query states
    query_states = model.q_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)

    # Compute value states
    value_states = model.v_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)

    seq_len = q_len

    # If past_key_value is not None, update the sequence length
    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            # for transformers <= 4.35
            seq_len += past_key_value[0].shape[-2]
        else:
            # since transformers 4.36, this is a DynamicCache instance
            seq_len += past_key_value.get_seq_length(model.layer_idx)

    # For transformers > 4.37.2 `position_ids` became a required arguments in the rotary embedding's forward pass.
    if "position_ids" not in inspect.signature(model.rotary_emb.forward).parameters:
        # TODO we assume that position_ids is not None here, not sure if that is safe but the old code also did that
        cos, sin = model.rotary_emb(value_states, seq_len=seq_len)
        return llama_apply_rotary_pos_emb(query_states, cos, sin, position_ids)

    past_seen_tokens = 0

    # Compute position_ids if they are not provided
    if position_ids is None:
        if past_key_value is None:
            new
