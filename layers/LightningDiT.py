"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from scipy.sparse.linalg import eigs
import networkx as nx
from typing import Optional, Union

from timm.models.vision_transformer import Mlp
from .swiglu_ffn import SwiGLUFFN 
from .RoPE import RotaryEmbedding
from .RMSNorm import RMSNorm


class NoisyTopKRouter(nn.Module):
    """
    Noisy Top-K Gating mechanism for Mixture-of-Experts.

    Args:
        input_size (int): Size of the input feature (hidden_dim).
        num_experts (int): Total number of experts.
        k (int): Number of experts to route each token to (default: 1, i.e., Switch).
        noise_epsilon (float, optional): Scaling factor for the noise. Defaults to 1e-2.
        capacity_factor (float, optional): Factor to scale expert capacity. >1.0 for buffer. Defaults to 1.25.
        dropout_rate (float, optional): Dropout rate applied to the gating logits. Defaults to 0.0.
    """

    def __init__(self, input_size, num_experts, k=1, noise_epsilon=1e-2, capacity_factor=1.25, dropout_rate=0.0):
        super(NoisyTopKRouter, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.k = k
        self.noise_epsilon = noise_epsilon
        self.capacity_factor = capacity_factor
        self.dropout_rate = dropout_rate

        # Gating linear layer: maps input to logits for each expert
        self.w_g = nn.Linear(input_size, num_experts, bias=False)
        # Noise linear layer: maps input to a scale for each expert's noise
        self.w_noise = nn.Linear(input_size, num_experts, bias=False)
        # Dropout layer for gating logits
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize weights appropriately
        nn.init.normal_(self.w_g.weight, mean=0.0, std=1.0 / (input_size ** 0.5))
        nn.init.normal_(self.w_noise.weight, mean=0.0, std=1.0 / (input_size ** 0.5))
        return

    def forward(self, x):
        """
        Forward pass of the router.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim].
            train (bool): Whether in training mode (for adding noise).

        Returns:
            gates (Tensor): Final gating weights of shape [batch_size*seq_len, num_experts].
                Non-top-k values are zero.
            indices (Tensor): The expert indices to route to for each token.
                Shape: [batch_size*seq_len, k]
            capacity (int): The calculated expert capacity.
            load (Tensor): The load balancing auxiliary loss value (scalar tensor).
        """
        # Flatten the batch and sequence dimensions
        # x shape: [batch_size, seq_len, hidden_dim] -> [n_tokens, hidden_dim]
        n_tokens = x.shape[0] * x.shape[1]
        x_flat = x.reshape(-1, self.input_size)

        # 1. Calculate base gating logits
        # logits shape: [n_tokens, num_experts]
        logits = self.w_g(x_flat)

        if self.w_noise.training:
            # 2. Calculate and add noise for exploration
            # noise_logits shape: [n_tokens, num_experts]
            noise_logits = self.w_noise(x_flat)
            # Generate random noise from a standard normal distribution
            noise = torch.randn_like(noise_logits) * self.noise_epsilon
            # Use Softplus to ensure the noise scale is positive and smooth
            noisy_noise_logits = noise * F.softplus(noise_logits)
            # Add the scaled noise to the original logits
            noisy_logits = logits + noisy_noise_logits
        else:
            # During inference, no noise is added
            noisy_logits = logits

        # Optional: Apply dropout to the noisy logits for regularization
        noisy_logits = self.dropout(noisy_logits)

        # 3. Apply Top-K selection and Softmax
        top_k_logits, indices = noisy_logits.topk(self.k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        gates = F.softmax(sparse_logits, dim=-1) # shape: [n_tokens, num_experts]

        # 4. Calculate expert capacity (number of tokens each expert can handle)
        expert_capacity = (n_tokens / self.num_experts)
        expert_capacity = int(expert_capacity * self.capacity_factor)
        expert_capacity = max(expert_capacity, self.k)

        # 5. (Optional but crucial) Calculate auxiliary loss for load balancing
        if self.w_noise.training:
            importance = gates.sum(dim=0) # shape: [num_experts]
            
            load = F.one_hot(indices, num_classes=self.num_experts).float().sum(dim=1) # shape: [n_tokens, num_experts]
            load = load.sum(dim=0) # shape: [num_experts] (count of how many tokens selected each expert)

            importance_loss = (importance.std() / (importance.mean() + 1e-10)) ** 2
            load_loss = (load.std() / (load.mean() + 1e-10)) ** 2
            aux_loss = importance_loss + load_loss
        else:
            aux_loss = torch.tensor(0.0, device=x.device) # No aux loss during inference

        return gates, indices, expert_capacity, aux_loss


class MoeEmbedder(nn.Module):
    embedding_list = {
        "Linear": nn.Linear,
        "MLP": Mlp,
    }
    
    def __init__(
        self, input_size, hidden_size, expert_nums=4, top_k=1,         
        expert_noise_epsilon=1e-2, 
        expert_capacity_factor=1.25, 
        expert_dropout_rate=0.0,
        embedding_type='Linear',
        residual_embedding=False, 
        **kwargs
    ):
        super().__init__()
        self.expert_nums = expert_nums
        self.top_k = top_k  # top-k experts
        self.input_size, self.hidden_size = input_size, hidden_size
        self.embedding_type = embedding_type
        self.residual_embedding = residual_embedding

        if self.expert_nums > 1:
            self.router = NoisyTopKRouter(
                            input_size = hidden_size,
                            num_experts = expert_nums,
                            k = top_k,
                            noise_epsilon = expert_noise_epsilon,
                            capacity_factor = expert_capacity_factor,
                            dropout_rate = expert_dropout_rate
                        )
            
            if "use_norm" not in kwargs:
                norm_layer_1 = nn.Identity()
                norm_layer_2 = nn.Identity()
            else:
                norm_layer_1 = RMSNorm(hidden_size) if kwargs["use_norm"] == "rmsnorm" \
                            else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                norm_layer_2 = RMSNorm(hidden_size) if kwargs["use_norm"] == "rmsnorm" \
                            else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

            
            if embedding_type == 'Linear':
                self.embedder_1 = nn.Linear(input_size, hidden_size)
                self.embedder_2 = nn.Sequential(
                    norm_layer_1,
                    nn.GELU(approximate="tanh"),
                )
                self.embedder_3 = norm_layer_2
                self.experts = nn.ModuleList(
                                    [nn.Linear(hidden_size, hidden_size) 
                                    for _ in range(self.expert_nums)]
                                )
            
            elif embedding_type == 'MLP':
                num_heads = kwargs["num_heads"]   
                self.embedder_1 = nn.Linear(input_size, hidden_size) 
                self.embedder_2 = nn.Sequential(
                    norm_layer_1,
                    Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
                )
                self.embedder_3 = norm_layer_2
                mlp_ratio = kwargs["mlp_ratio"]
                self.experts = nn.ModuleList(
                                    [Mlp(in_features=hidden_size, hidden_features=hidden_size*int(mlp_ratio), drop=0) 
                                    for _ in range(self.expert_nums)]
                                )
        else:
            raise NotImplementedError
        return
    
    
    def forward(self, x):
        if self.expert_nums > 1:         
            x = self.embedder_1(x)
            x_imd = x + self.embedder_2(x) if self.residual_embedding else self.embedder_2(x)
            x = self.embedder_3(x_imd)
            
            original_shape = x.shape
            gates, indices, expert_capacity, aux_loss = self.router(x)
            x_flat = x.reshape(-1, self.hidden_size)   # [batch_size * seq_len, hidden_size]
            n_tokens = x_flat.size(0)

            # 1. Dispatch tokens to experts
            expert_inputs = self._dispatch_to_experts(x_flat, indices, expert_capacity)
            # 2. Process tokens with experts
            expert_outputs = self._process_with_experts(expert_inputs)
            # 3. Combine expert outputs
            combined_output = self._combine_expert_outputs(x_flat, expert_outputs, indices, gates, 
                                                           expert_capacity, n_tokens, original_shape)
            if self.residual_embedding:
                y = x_imd + combined_output
            else:
                y = combined_output
        else:
            raise NotImplementedError
        return y, aux_loss


    def _dispatch_to_experts(self, x_flat, indices, expert_capacity):
        """
        Dispatch tokens to their assigned experts.

        Returns:
            expert_inputs: List of tensors, each of shape [expert_capacity, input_size]
        """
        # Create a mask for which tokens go to which experts
        # expert_mask: [n_tokens, num_experts] - binary mask
        expert_mask = F.one_hot(indices, num_classes=self.expert_nums).sum(dim=1).float()
        
        expert_inputs = []
        for expert_idx in range(self.expert_nums):
            # Find which tokens are assigned to this expert
            token_indices = torch.nonzero(expert_mask[:, expert_idx], as_tuple=True)[0]
            
            if len(token_indices) == 0:
                expert_input = torch.zeros((expert_capacity, self.hidden_size), device=x_flat.device, dtype=x_flat.dtype)
            else:
                tokens_for_expert = x_flat[token_indices]
                # If we have more tokens than capacity, we need to truncate
                if len(tokens_for_expert) > expert_capacity:
                    # In practice, you might want to handle this more gracefully
                    tokens_for_expert = tokens_for_expert[:expert_capacity]
                
                # Pad if we have fewer tokens than capacity
                if len(tokens_for_expert) < expert_capacity:
                    padding = torch.zeros((expert_capacity - len(tokens_for_expert), self.hidden_size),
                                        device=x_flat.device, dtype=x_flat.dtype)
                    tokens_for_expert = torch.cat([tokens_for_expert, padding], dim=0)
                expert_input = tokens_for_expert
            
            expert_inputs.append(expert_input)
        return expert_inputs


    def _process_with_experts(self, expert_inputs):
        """
        Process the dispatched tokens through each expert.
        """
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            expert_input = expert_inputs[expert_idx]
            # Process through expert
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)
        return expert_outputs


    def _combine_expert_outputs(self, x_flat, expert_outputs, indices, gates, expert_capacity, n_tokens, original_shape):
        """
        Combine the processed tokens from experts back together.
        """
        # Create empty output tensor
        combined_flat = torch.zeros((n_tokens, self.hidden_size), device=expert_outputs[0].device, dtype=expert_outputs[0].dtype)
        # Create count tensor to handle multiple experts per token
        count = torch.zeros((n_tokens,), device=combined_flat.device)
        
        # For each expert, scatter its outputs back to the original positions
        for expert_idx in range(self.expert_nums):
            # Find which tokens were processed by this expert
            expert_mask = (indices == expert_idx).any(dim=1)
            token_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
            
            if len(token_indices) > 0:
                # Get this expert's output (only the relevant part, not padding)
                expert_output = expert_outputs[expert_idx]
                actual_tokens = min(len(token_indices), expert_capacity)
                expert_output_actual = expert_output[:actual_tokens]
                
                # Get the gate weights for these tokens
                token_gates = gates[token_indices[:actual_tokens], expert_idx]
                
                # Add weighted contributions
                combined_flat[token_indices[:actual_tokens]] += expert_output_actual * token_gates.unsqueeze(1)
                count[token_indices[:actual_tokens]] += token_gates
        
        # Handle tokens that might not have been processed (due to capacity issues)
        # For these tokens, we use the original input (zero expert contribution)
        unprocessed_mask = (count == 0)
        if unprocessed_mask.any():
            # Use original input for unprocessed tokens (pass-through)
            combined_flat[unprocessed_mask] = x_flat[unprocessed_mask]
        
        # Reshape back to original shape
        combined = combined_flat.view(original_shape)
        return combined   
        
        
        
def modulate(x, shift, scale):
    if x.ndim == scale.ndim:
        if shift is None:
            return x * (1 + scale)
        return x * (1 + scale) + shift
    else:
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = False,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        return
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    CrossAttention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        c_dim: int = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = False,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        # If c_dim is not provided, use dim (self-attention like)
        self.c_dim = c_dim if c_dim is not None else dim
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        # Query comes from input x, Key and Value come from context
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(self.c_dim, dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        return
        
    def forward(self, x: torch.Tensor, c: torch.Tensor = None, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        # If no context is provided, use x as context (self-attention fallback)
        if c is None:
            c = x
                    
        # Query from input x
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Key and Value from context
        kv = self.kv(c).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        if len(t.shape) < 1:
            t = torch.reshape(t, shape=(-1,))
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        context_fusion_type='adaLN',
        expert_nums=1,   
        top_k=1,
        expert_noise_epsilon=1e-2, 
        expert_capacity_factor=1.25, 
        expert_dropout_rate=0.0,
        **block_kwargs
    ):
        super().__init__()
                
        self.hidden_size = hidden_size
        self.num_heads, self.mlp_ratio = num_heads, mlp_ratio
        
        self.use_qknorm = use_qknorm
        self.use_swiglu = use_swiglu 
        self.use_rmsnorm = use_rmsnorm
        self.wo_shift = wo_shift

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer and MoE
        self.expert_nums = expert_nums   
        self.top_k = top_k
        self.expert_noise_epsilon=1e-2, 
        self.expert_capacity_factor=1.25, 
        self.expert_dropout_rate=0.0,
        
        def _build_ffn_layer():
            # Initialize MLP layer
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            if use_swiglu:
                # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
                mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
            else:
                mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            return mlp       
        
        if self.expert_nums > 1:
            self.router = NoisyTopKRouter(
                            input_size = hidden_size,
                            num_experts = expert_nums,
                            k = top_k,
                            noise_epsilon = expert_noise_epsilon,
                            capacity_factor = expert_capacity_factor,
                            dropout_rate = expert_dropout_rate
                        )
            self.experts = nn.ModuleList([_build_ffn_layer() for _ in range(self.expert_nums)])
        else:
            self.mlp = _build_ffn_layer()

        self.context_fusion_type = context_fusion_type
        if context_fusion_type == 'crossAttn':
            self._build_cross_attn_version(**block_kwargs)
        elif context_fusion_type == 'adaLN':
            self._build_adaLN_version()
        elif context_fusion_type == 'adaLN+crossAttn':
            self._build_adaLNplusCrossAttn_version()
        else:
            raise NotImplementedError
        return


    def _build_cross_attn_version(self, **block_kwargs):
        if not self.use_rmsnorm:
            self.norm3 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm4 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm3 = RMSNorm(self.hidden_size)
            self.norm4 = RMSNorm(self.hidden_size)
        # Initialize attention layer
        self.cross_attn = CrossAttention(self.hidden_size, self.hidden_size, num_heads=self.num_heads, 
                                         qkv_bias=True, qk_norm=self.use_qknorm, use_rmsnorm=self.use_rmsnorm,
                                         **block_kwargs)    
        return       
    
    
    def _build_adaLN_version(self):
        # Initialize AdaLN modulation
        if self.wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
            )
        return
    

    def _build_adaLNplusCrossAttn_version(self):
        self._build_adaLN_version()
        self._build_cross_attn_version()
        return


    def forward(self, x, c, feat_rope=None):
        if self.context_fusion_type == 'crossAttn':
            x, aux_loss = self.foward_cross_attn(x, c, feat_rope)
        elif self.context_fusion_type == 'adaLN':
            x, aux_loss = self.foward_adaln(x, c, feat_rope)
        elif self.context_fusion_type == 'adaLN+crossAttn':
            x, aux_loss = self.forward_adaln_plus_cross_attn(x, c, feat_rope)
        else:
            raise NotImplementedError        
        return x, aux_loss
    
    
    def foward_adaln(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        if x.ndim != scale_msa.ndim:
            scale_msa, gate_msa, scale_mlp, gate_mlp = scale_msa.unsqueeze(1), gate_msa.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)
            if shift_msa is not None:
                shift_msa, shift_mlp = shift_msa.unsqueeze(1), shift_mlp.unsqueeze(1)
            
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        
        if self.expert_nums > 1:
            x = modulate(self.norm2(x), shift_mlp, scale_mlp)   ### 20250923前忘记这一句, 所以之前MoE结果不准确, 虽然也几乎没用过
            
            original_shape = x.shape
            gates, indices, expert_capacity, aux_loss = self.router(x)
            x_flat = x.view(-1, self.hidden_size)   # [batch_size * seq_len, hidden_size]
            n_tokens = x_flat.size(0)

            # 1. Dispatch tokens to experts
            expert_inputs = self._dispatch_to_experts(x_flat, indices, expert_capacity)
            # 2. Process tokens with experts
            expert_outputs = self._process_with_experts(expert_inputs)
            # 3. Combine expert outputs
            combined_output = self._combine_expert_outputs(x_flat, expert_outputs, indices, gates, 
                                                           expert_capacity, n_tokens, original_shape)
            x = x + gate_mlp * combined_output
        else:
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            aux_loss = torch.tensor(0.0, device=x.device)
        return x, aux_loss
    
    
    def foward_cross_attn(self, x, c, feat_rope=None):            
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.cross_attn(self.norm2(x), self.norm3(c), rope=feat_rope)
        
        if self.expert_nums > 1:
            x = self.norm4(x)
            
            original_shape = x.shape
            gates, indices, expert_capacity, aux_loss = self.router(x)
            x_flat = x.view(-1, self.hidden_size)   # [batch_size * seq_len, hidden_size]
            n_tokens = x_flat.size(0)

            # 1. Dispatch tokens to experts
            expert_inputs = self._dispatch_to_experts(x_flat, indices, expert_capacity)
            # 2. Process tokens with experts
            expert_outputs = self._process_with_experts(expert_inputs)
            # 3. Combine expert outputs
            combined_output = self._combine_expert_outputs(x_flat, expert_outputs, indices, gates, 
                                                           expert_capacity, n_tokens, original_shape)
            x = x + combined_output
        else:
            x = x + self.mlp(self.norm4(x))
            aux_loss = torch.tensor(0.0, device=x.device)
        return x, aux_loss
    
    
    def forward_adaln_plus_cross_attn(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        if x.ndim != scale_msa.ndim:
            scale_msa, gate_msa, scale_mlp, gate_mlp = scale_msa.unsqueeze(1), gate_msa.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)
            if shift_msa is not None:
                shift_msa, shift_mlp = shift_msa.unsqueeze(1), shift_mlp.unsqueeze(1)
            
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + self.cross_attn(self.norm2(x), self.norm3(c), rope=feat_rope)

        if x.ndim != scale_msa.ndim:
            scale_msa, gate_msa, scale_mlp, gate_mlp = scale_msa.unsqueeze(1), gate_msa.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)
            if shift_msa is not None:
                shift_msa, shift_mlp = shift_msa.unsqueeze(1), shift_mlp.unsqueeze(1)

        if self.expert_nums > 1:
            x = modulate(self.norm4(x), shift_mlp, scale_mlp)   ### 20250923前忘记这一句, 所以之前MoE结果不准确, 虽然也几乎没用过
            
            original_shape = x.shape
            gates, indices, expert_capacity, aux_loss = self.router(x)
            x_flat = x.view(-1, self.hidden_size)   # [batch_size * seq_len, hidden_size]
            n_tokens = x_flat.size(0)

            # 1. Dispatch tokens to experts
            expert_inputs = self._dispatch_to_experts(x_flat, indices, expert_capacity)
            # 2. Process tokens with experts
            expert_outputs = self._process_with_experts(expert_inputs)
            # 3. Combine expert outputs
            combined_output = self._combine_expert_outputs(x_flat, expert_outputs, indices, gates, 
                                                           expert_capacity, n_tokens, original_shape)
            x = x + gate_mlp * combined_output
        else:
            x = x + gate_mlp * self.mlp(modulate(self.norm4(x), shift_mlp, scale_mlp))
            aux_loss = torch.tensor(0.0, device=x.device)
        return x, aux_loss
    

    def _dispatch_to_experts(self, x_flat, indices, expert_capacity):
        """
        Dispatch tokens to their assigned experts.

        Returns:
            expert_inputs: List of tensors, each of shape [expert_capacity, input_size]
        """
        # Create a mask for which tokens go to which experts
        # expert_mask: [n_tokens, num_experts] - binary mask
        expert_mask = F.one_hot(indices, num_classes=self.expert_nums).sum(dim=1).float()
        
        expert_inputs = []
        for expert_idx in range(self.expert_nums):
            # Find which tokens are assigned to this expert
            token_indices = torch.nonzero(expert_mask[:, expert_idx], as_tuple=True)[0]
            
            if len(token_indices) == 0:
                expert_input = torch.zeros((expert_capacity, self.hidden_size), device=x_flat.device, dtype=x_flat.dtype)
            else:
                tokens_for_expert = x_flat[token_indices]
                # If we have more tokens than capacity, we need to truncate
                if len(tokens_for_expert) > expert_capacity:
                    # In practice, you might want to handle this more gracefully
                    tokens_for_expert = tokens_for_expert[:expert_capacity]
                
                # Pad if we have fewer tokens than capacity
                if len(tokens_for_expert) < expert_capacity:
                    padding = torch.zeros((expert_capacity - len(tokens_for_expert), self.hidden_size),
                                        device=x_flat.device, dtype=x_flat.dtype)
                    tokens_for_expert = torch.cat([tokens_for_expert, padding], dim=0)
                expert_input = tokens_for_expert
            
            expert_inputs.append(expert_input)
        return expert_inputs


    def _process_with_experts(self, expert_inputs):
        """
        Process the dispatched tokens through each expert.
        """
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            expert_input = expert_inputs[expert_idx]
            # Process through expert
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)
        return expert_outputs


    def _combine_expert_outputs(self, x_flat, expert_outputs, indices, gates, expert_capacity, n_tokens, original_shape):
        """
        Combine the processed tokens from experts back together.
        """
        # Create empty output tensor
        combined_flat = torch.zeros((n_tokens, self.hidden_size), device=expert_outputs[0].device, dtype=expert_outputs[0].dtype)
        # Create count tensor to handle multiple experts per token
        count = torch.zeros((n_tokens,), device=combined_flat.device)
        
        # For each expert, scatter its outputs back to the original positions
        for expert_idx in range(self.expert_nums):
            # Find which tokens were processed by this expert
            expert_mask = (indices == expert_idx).any(dim=1)
            token_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
            
            if len(token_indices) > 0:
                # Get this expert's output (only the relevant part, not padding)
                expert_output = expert_outputs[expert_idx]
                actual_tokens = min(len(token_indices), expert_capacity)
                expert_output_actual = expert_output[:actual_tokens]
                
                # Get the gate weights for these tokens
                token_gates = gates[token_indices[:actual_tokens], expert_idx]
                
                # Add weighted contributions
                combined_flat[token_indices[:actual_tokens]] += expert_output_actual * token_gates.unsqueeze(1)
                count[token_indices[:actual_tokens]] += token_gates
        
        # Handle tokens that might not have been processed (due to capacity issues)
        # For these tokens, we use the original input (zero expert contribution)
        unprocessed_mask = (count == 0)
        if unprocessed_mask.any():
            # Use original input for unprocessed tokens (pass-through)
            combined_flat[unprocessed_mask] = x_flat[unprocessed_mask]
        
        # Reshape back to original shape
        combined = combined_flat.view(original_shape)
        return combined    
    


class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        length = 32,
        node_n = 18,
        in_channels = 4,
        hidden_size = 1152,
        depth = 28,
        num_heads = 16,
        mlp_ratio = 4.0,
        class_dropout_prob = 0.1,
        num_classes = 16,
        learn_sigma = False,
        use_qknorm = False,
        use_swiglu = False,
        use_rope = False,
        use_rmsnorm = False,
        wo_shift = False,
        context_fusion_type = 'adaLN',
        use_graph_pos_embed = False,
        use_checkpoint = False,
        expert_nums=1,   
        top_k=1,
        expert_noise_epsilon=1e-2, 
        expert_capacity_factor=1.25, 
        expert_dropout_rate=0.0,
        **kwargs
    ):        
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.length = length
        self.node_n = node_n
        self.num_heads = num_heads
        
        self.use_qknorm = use_qknorm
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.wo_shift = wo_shift
        self.use_checkpoint = use_checkpoint
        
        self.depth = depth
        self.hidden_size = hidden_size
        
        self.model_in_channels = self.in_channels * self.node_n 
        self.model_out_channels = self.out_channels * self.node_n

        self.x_embedder = nn.Linear(self.model_in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if "cond_in_channels" in kwargs:
            self.cond_in_channels = kwargs["cond_in_channels"]
            if self.cond_in_channels > 0:
                if "cond_expert_nums" in kwargs and kwargs["cond_expert_nums"] > 1:
                    self.cond_embedder = MoeEmbedder(
                                            self.cond_in_channels, hidden_size,
                                            expert_nums = kwargs["cond_expert_nums"], 
                                            top_k = kwargs["cond_top_k"],
                                            embedding_type = kwargs["cond_embedding_type"],
                                            residual_embedding = kwargs["residual_embedding"],
                                            expert_noise_epsilon = kwargs["cond_expert_noise_epsilon"], 
                                            expert_capacity_factor = kwargs["cond_expert_capacity_factor"],
                                            expert_dropout_rate = kwargs["cond_expert_dropout_rate"], 
                                            **kwargs["cond_embedding_params"]
                                        )
                else:
                    self.cond_embedder = nn.Linear(self.cond_in_channels, hidden_size, bias=True)
                    
        # Will use fixed sin-cos embedding:
        self.use_graph_pos_embed = use_graph_pos_embed
        if use_graph_pos_embed:
            laplacian_pe = LaplacianPositionalEncoding(hidden_size)
            self.pos_embed = laplacian_pe(kwargs['adj_matrix'])
            self.pos_embed = torch.repeat_interleave(self.pos_embed, repeats=self.length // self.pos_embed.shape[0], dim=0)
            self.pos_embed = nn.Parameter(self.pos_embed[None, ...], requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.length, hidden_size), requires_grad=False)
            
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads
            hw_seq_len = self.length
            self.feat_rope = RotaryEmbedding(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.context_fusion_type = context_fusion_type
        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     context_fusion_type=context_fusion_type,
                     expert_nums=expert_nums,   
                     top_k=top_k,
                     expert_noise_epsilon=expert_noise_epsilon, 
                     expert_capacity_factor=expert_capacity_factor, 
                     expert_dropout_rate=expert_dropout_rate,
                    ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, 1, self.model_out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()
        return
    

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if not self.use_graph_pos_embed:
            pos_embed = sincos_pos_embed(self.length, self.pos_embed.shape[-1])
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            if self.context_fusion_type == "adaLN" or self.context_fusion_type == "adaLN+crossAttn":
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        return


    def forward(self, x, t, **kwargs):
        """
        Forward pass of LightningDiT.
        x: (T, B, C, N) tensor of spatial inputs (poses or latent representations of poses)
        t: (B,) tensor of diffusion timesteps
        use_checkpoint: boolean to toggle checkpointing
        """
        T, B, C, N = x.shape
        x = torch.swapaxes(x, 0, 1)
        x = torch.reshape(x, shape=(B, T, C*N))

        x = self.x_embedder(x) + self.pos_embed  # (B, T, C)
        t = self.t_embedder(t)    

        aux_loss_list = []
        if "x_cond" in kwargs and hasattr(self, "cond_embedder"):
            x_cond = kwargs["x_cond"]
            x_cond = torch.swapaxes(x_cond, 0, 1)
            x_cond = torch.reshape(x_cond, shape=(x_cond.shape[0], x_cond.shape[1], -1))
            x_cond = self.cond_embedder(x_cond)
            if isinstance(x_cond, tuple):
                x_cond, aux_loss = x_cond
                aux_loss_list.append(aux_loss.reshape(1,))
            if hasattr(self, "cond_as_input"):
                x = x + x_cond
            elif hasattr(self, "cond_everywhere"):
                x = x + x_cond
                t = t[:, None, :].expand(-1, x_cond.shape[1], -1) + x_cond
            else:
                t = t[:, None, :].expand(-1, x_cond.shape[1], -1) + x_cond
        
        c = t
        for block in self.blocks:
            if self.use_checkpoint:
                x, aux_loss = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x, aux_loss = block(x, c, self.feat_rope)     # (B, T, C)            
            aux_loss_list.append(aux_loss.reshape(1,))
        x = self.final_layer(x, c)      # (B, T, out_channels)          

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        
        x = torch.reshape(x, shape=(B, T, C, N))
        x = torch.swapaxes(x, 0, 1)
        mean_aux_loss = torch.mean(torch.concat(aux_loss_list), dim=0)
        
        if self.final_layer.training:
            return x, mean_aux_loss
        else:
            return x




def get_2d_sincos_pos_embed(embed_dim, grid_size:tuple, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def sincos_pos_embed(max_len, d_model):
    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model/2,)
    
    pe = np.zeros((max_len, d_model))  # 初始化位置编码矩阵
    sin_val = np.sin(position * div_term)  # (max_len, d_model//2)
    cos_val = np.cos(position * div_term)  # (max_len, d_model//2)
    # 交替填充sin和cos
    pe[:, 0::2] = sin_val[:, :(d_model + 1) // 2]  # 所有sin值
    pe[:, 1::2] = cos_val[:, :d_model // 2]       # 所有cos值
    return pe



class LaplacianPositionalEncoding(nn.Module):
    """
    基于拉普拉斯特征向量的图位置编码
    """
    def __init__(self, embedding_dim: int, normalization: str = 'sym'):
        """
        Args:
            embedding_dim: 位置编码的维度
            normalization: 拉普拉斯矩阵归一化方式 ('sym', 'rw', None)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalization = normalization
        return
        
    def compute_laplacian(self, adj_matrix: np.ndarray) -> np.ndarray:
        """计算拉普拉斯矩阵"""
        # 确保是对称矩阵（无向图）
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        # 度矩阵
        degree = np.diag(adj_matrix.sum(axis=1))
        if self.normalization is None:
            # 非归一化拉普拉斯矩阵: L = D - A
            laplacian = degree - adj_matrix
        elif self.normalization == 'sym':
            # 对称归一化拉普拉斯矩阵: L = I - D^(-1/2) A D^(-1/2)
            degree_inv_sqrt = np.diag(1.0 / np.sqrt(adj_matrix.sum(axis=1)))
            laplacian = np.eye(adj_matrix.shape[0]) - degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt
        elif self.normalization == 'rw':
            # 随机游走归一化拉普拉斯矩阵: L = I - D^(-1) A
            degree_inv = np.diag(1.0 / adj_matrix.sum(axis=1))
            laplacian = np.eye(adj_matrix.shape[0]) - degree_inv @ adj_matrix
        else:
            raise ValueError(f"不支持的归一化方式: {self.normalization}")
        return laplacian
    
    def get_eigenvectors(self, laplacian: np.ndarray, k: int) -> np.ndarray:
        """计算拉普拉斯矩阵的特征向量"""
        # 使用scipy的eigs函数计算前k个最小特征值对应的特征向量
        # 注意：拉普拉斯矩阵的特征值是非负实数，最小的特征值是0
        eigenvalues, eigenvectors = eigs(laplacian, k=k, which='SR')  # SR: Smallest Real
        # 按特征值排序（从小到大）
        idx = np.argsort(np.real(eigenvalues))
        eigenvectors = np.real(eigenvectors[:, idx])
        
        return eigenvectors
    
    def forward(self, adj_matrix: Union[np.ndarray, torch.Tensor], 
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            adj_matrix: 邻接矩阵 (n_nodes, n_nodes)
            node_features: 可选的原始节点特征 (n_nodes, feature_dim)
            
        Returns:
            position_encoding: 位置编码 (n_nodes, embedding_dim)
        """
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()
        n_nodes = adj_matrix.shape[0]
        # 确保邻接矩阵是连通的
        if not self.is_connected(adj_matrix):
            print("警告: 图不是连通的，可能影响位置编码效果")
        
        # 计算拉普拉斯矩阵
        laplacian = self.compute_laplacian(adj_matrix)
        
        # 计算特征向量（去掉第一个特征值为0的特征向量）
        k = min(self.embedding_dim + 1, n_nodes - 1)
        if k <= 0:
            # 如果节点数太少，返回零向量
            return torch.zeros(n_nodes, self.embedding_dim)
            
        eigenvectors = self.get_eigenvectors(laplacian, k)
        
        # 去掉第一个特征向量（对应特征值0，包含图的连通分量信息）
        positional_encoding = eigenvectors[:, 1:self.embedding_dim+1]
        
        # 如果维度不够，用零填充
        if positional_encoding.shape[1] < self.embedding_dim:
            padding = np.zeros((n_nodes, self.embedding_dim - positional_encoding.shape[1]))
            positional_encoding = np.hstack([positional_encoding, padding])
        
        # 转换为torch tensor并归一化
        positional_encoding = torch.FloatTensor(positional_encoding)
        positional_encoding = self.normalize_encoding(positional_encoding)
        return positional_encoding
    
    def is_connected(self, adj_matrix: np.ndarray) -> bool:
        """检查图是否连通"""
        G = nx.from_numpy_array(adj_matrix)
        return nx.is_connected(G)
    
    def normalize_encoding(self, encoding: torch.Tensor) -> torch.Tensor:
        """对位置编码进行归一化"""
        # 按行归一化到[-1, 1]范围
        encoding = encoding - encoding.mean(dim=0, keepdim=True)
        std = encoding.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        encoding = encoding / std
        return encoding



'''
#################################################################################
#                             LightningDiT Configs                              #
#################################################################################

def LightningDiT_XL_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def LightningDiT_XL_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_L_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_B_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def LightningDiT_B_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def LightningDiT_1p0B_1(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)

def LightningDiT_1p0B_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

def LightningDiT_1p6B_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)

def LightningDiT_1p6B_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)

LightningDiT_models = {
    'LightningDiT-B/1': LightningDiT_B_1, 'LightningDiT-B/2': LightningDiT_B_2,
    'LightningDiT-L/2': LightningDiT_L_2,
    'LightningDiT-XL/1': LightningDiT_XL_1, 'LightningDiT-XL/2': LightningDiT_XL_2,
    'LightningDiT-1p0B/1': LightningDiT_1p0B_1, 'LightningDiT-1p0B/2': LightningDiT_1p0B_2,
    'LightningDiT-1p6B/1': LightningDiT_1p6B_1, 'LightningDiT-1p6B/2': LightningDiT_1p6B_2,
}
'''