import math
import os
import json
import tensorflow as tf

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_download import load_gpt2_params_from_tf_ckpt

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


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
        
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
            for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
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
        logits = self.out_head(x)
        return logits
    

class OscarNomGPT(nn.Module):
    def __init__(self, config, gpt_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # GPT-based encoder
        self.chunk_size = config["context_length"]
        self.enc_d_model = config["emb_dim"]

        self.encoder = GPTModel(config)
        if gpt_params:
            self.encoder = self._load_weights_into_gpt(self.encoder, gpt_params)

        # TODO: Freeze GPT weights

        # TODO: replace last linear layer with this one somehow
        self.chunk_head = nn.Linear(config["emb_dim"], config["emb_dim"])

        if config["emb_dim"] != config['agg_d_model']:
            self.chunk_proj = nn.Linear(config["emb_dim"], config['agg_d_model'])
        else:
            self.chunk_proj = nn.Identity()

        # aggregator
        self.agg_d_model = config['agg_d_model']
        self.agg_nhead = config['agg_nhead']
        self.agg_dim_ff = config['agg_dim_ff']
        self.agg_num_layers = config['agg_num_layers']

        self.agg_pos_enc = self._positional_encoder(config['max_seq_len'] // config["context_length"] + 1, config['agg_d_model'])

        aggregator_layer = nn.TransformerEncoderLayer(
            d_model=config['agg_d_model'],
            nhead=config['agg_nhead'],
            dim_feedforward=config['agg_dim_ff'],
            dropout=config["drop_rate"],
            batch_first=True
        )
        self.aggregator = nn.TransformerEncoder(aggregator_layer, num_layers=config['agg_num_layers'])

        self.dropout= nn.Dropout(config["drop_rate"])

        self.classification_head = nn.Linear(config['agg_d_model'], 2)
    
    def _load_weights_into_gpt(self, gpt, params):
        def assign(left, right):
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                                "Right: {right.shape}"
                )
            return torch.nn.Parameter(torch.tensor(right))
        
        # set token and positional embedding weights
        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])): # for each transformer block

            # np.split divides weights into three equal parts for query, key, and value components
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            
            # Q, K, V weights
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)
            
            # Q, K, V biases
            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)
            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])
            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])
            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])
            
        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

        # weight tying
        # re-use weights of the token embedding layer now in the output layer
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
        return gpt

    def _positional_encoder(self, max_seq_len, d_model):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, src):
        batch_size, seq_len = src.shape

        # 1. Chunk the input
        # Pad seq_len to be divisible by chunk_size
        remainder = seq_len % self.chunk_size
        if remainder != 0 :
            pad_len = self.chunk_size - remainder
            src = F.pad(src, (0, pad_len), value=0)
            seq_len = src.shape[1]

        num_chunks = seq_len // self.chunk_size

        # Then reshape to (batch_size * num_chunks, chunk_size)
        src = src.view((batch_size, num_chunks, self.chunk_size))
        src = src.view((batch_size * num_chunks, self.chunk_size))

        # 2. GPT-2 encodes the input into embedings
        # Shape after encoder: (batch_size * num_chunks, chunk_size, enc_d_model)
        enc_chunks = self.encoder(src)

        # 3. Pool each chunk to single vector (mean pool over token dimension)
        # Shape: (batch_size * num_chunks, enc_d_model)
        chunk_embs = enc_chunks.mean(dim=1)

        # 4. Reshape back to (batch_size, num_chunks, enc_d_model)
        chunk_embs = chunk_embs.view((batch_size,num_chunks, self.enc_d_model))

        # 5. Project if needed and add chunk positional encoding
        # Shape: (batch_size, num_chunks, agg_d_model)
        chunk_embs = self.chunk_proj(chunk_embs) * math.sqrt(self.agg_d_model)
        chunk_embs += self.agg_pos_enc[:, :num_chunks, :].to(chunk_embs.device)
        chunk_embs = self.dropout(chunk_embs)

        # 6. Run through aggregator transformer
        # Shape (batch_size, num_chunks, agg_d_model)
        agg_out = self.aggregator(chunk_embs)

        # 7. Pool to single vector (mean pool over chunk dimension)
        # Shape: (batch_size, agg_d_model)
        agg_out = agg_out.mean(dim=1)

        # 8. Classification head
        # Shape: (batch_size, 2)
        logits = self.classification_head(agg_out)

        return logits

if __name__ == "__main__":
    torch.manual_seed(1337)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_size="124M"
    models_dir="gpt2"

    # Define GPT model path
    model_dir = os.path.join("src", models_dir, model_size)

    # Load GPT settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,

        'agg_d_model': 64,
        'agg_nhead': 2,
        'agg_dim_ff': 128,
        'agg_num_layers': 1,
        
        'max_seq_len': 106578,

        "drop_rate": 0.1,
        "qkv_bias": True
    }

    model = OscarNomGPT(config, params).to(device)