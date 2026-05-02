import math
import os
import json
import tensorflow as tf

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # compute output size of each head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

        self.out_proj = nn.Linear(d_out, d_out) # linear layer combines head outputs

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys       = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values   = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys       = keys.transpose(1, 2) # (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values   = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # (b, num_tokens, n_heads, head_dim)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # combine heads

        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # in case var is actually 0, add small eps to prevent division by 0

        # in case model performance improves without layer norm, 
        # scale and shift parameters will change significantly during training
        self.scale = nn.Parameter(torch.ones(emb_dim)) 
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        # unbiased=False to keep computations consistent with original GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # arbitrary multiply by factor of 4
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        return x
        
class GPTEmbedModel(nn.Module):
    def __init__(self, cfg, params=None):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
            for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # still needed for easy loading of GPT-2 pretrained weights
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        if params:
            self._load_weights_into_gpt(params)
            
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # skipped since another module only needs the emb_dim sized output
        # logits = self.out_head(x)
        return x
    
    def _load_weights_into_gpt(self, params):
        def assign(left, right):
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                                "Right: {right.shape}"
                )
            return torch.nn.Parameter(torch.tensor(right))
        
        # set token and positional embedding weights
        self.pos_emb.weight = assign(self.pos_emb.weight, params['wpe'])
        self.tok_emb.weight = assign(self.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])): # for each transformer block

            # np.split divides weights into three equal parts for query, key, and value components
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            
            # Q, K, V weights
            self.trf_blocks[b].att.W_query.weight = assign(
                self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = assign(
                self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = assign(
                self.trf_blocks[b].att.W_value.weight, v_w.T)
            
            # Q, K, V biases
            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = assign(
                self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = assign(
                self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = assign(
                self.trf_blocks[b].att.W_value.bias, v_b)
            self.trf_blocks[b].att.out_proj.weight = assign(
                self.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.trf_blocks[b].att.out_proj.bias = assign(
                self.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])
            self.trf_blocks[b].ff.layers[0].weight = assign(
                self.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.trf_blocks[b].ff.layers[0].bias = assign(
                self.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.trf_blocks[b].ff.layers[2].weight = assign(
                self.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            self.trf_blocks[b].ff.layers[2].bias = assign(
                self.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])
            self.trf_blocks[b].norm1.scale = assign(
                self.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            self.trf_blocks[b].norm1.shift = assign(
                self.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            self.trf_blocks[b].norm2.scale = assign(
                self.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            self.trf_blocks[b].norm2.shift = assign(
                self.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])
            
        self.final_norm.scale = assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = assign(self.final_norm.shift, params["b"])

        # weight tying
        # re-use weights of the token embedding layer now in the output layer
        self.out_head.weight = assign(self.out_head.weight, params["wte"])
        return self