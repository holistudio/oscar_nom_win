import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            cfg["emb_dim"], cfg["enc_d_model"], bias=False
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
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # GPT-based encoder
        self.chunk_size = config["context_length"]
        self.enc_d_model = config["enc_d_model"]
        self.micro_batch_size = config.get("micro_batch_size", 4)

        self.encoder = GPTModel(config)

        # Freeze the encoder — we only train the classification head
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(config['drop_rate'])

        self.classification_head = nn.Linear(config["enc_d_model"], 2)
    
    def forward(self, src):
        batch_size, seq_len = src.shape

        # 1. Chunk the input
        # Pad seq_len to be divisible by chunk_size
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            src = F.pad(src, (0, pad_len), value=0)
            seq_len = src.shape[1]

        num_chunks = seq_len // self.chunk_size

        # Reshape to (batch_size, num_chunks, chunk_size)
        src = src.view(batch_size, num_chunks, self.chunk_size)

        # 2. Encode chunks in micro-batches under no_grad (encoder is frozen)
        #    This is the critical fix: no_grad tells PyTorch not to store
        #    activations for backprop, so memory from each micro-batch is
        #    freed before the next one starts.
        all_chunk_embs = []

        with torch.no_grad():
            for b_idx in range(batch_size):
                chunks_for_sample = src[b_idx]  # (num_chunks, chunk_size)

                sample_chunk_embs = []
                for i in range(0, num_chunks, self.micro_batch_size):
                    micro_batch = chunks_for_sample[i : i + self.micro_batch_size]  # (mb_size, chunk_size)

                    # GPT-2 encodes the micro-batch
                    # Shape: (mb_size, chunk_size, enc_d_model)
                    enc_out = self.encoder(micro_batch)

                    # Mean pool each chunk over the token dimension
                    # Shape: (mb_size, enc_d_model)
                    pooled = enc_out.mean(dim=1)

                    sample_chunk_embs.append(pooled)

                # Stack all chunk embeddings for this sample
                # Shape: (num_chunks, enc_d_model)
                sample_chunk_embs = torch.cat(sample_chunk_embs, dim=0)
                all_chunk_embs.append(sample_chunk_embs)

        # Shape: (batch_size, num_chunks, enc_d_model)
        # .detach() is redundant with no_grad but makes the intent explicit:
        # the classification head is the only thing that gets gradients
        chunk_embs = torch.stack(all_chunk_embs, dim=0).detach()

        # 3. Aggregate across chunks (mean pool over chunk dimension)
        # Shape: (batch_size, enc_d_model)
        agg_out = chunk_embs.mean(dim=1)

        agg_out = self.dropout(agg_out)

        # 4. Classification head
        # Shape: (batch_size, 2)
        logits = self.classification_head(agg_out)

        return logits

if __name__ == '__main__':
    torch.manual_seed(1337)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = {
        'context_length': 1024,
        'vocab_size': 50257,
        'emb_dim': 768,
        'enc_d_model': 256,
        'n_heads': 12,
        'n_layers': 12,

        'max_seq_len': 106578,
        'micro_batch_size': 4,

        "drop_rate": 0.1,
        "qkv_bias": True
    }

    model = OscarNomGPT(config).to(device)

    batch_size = 1
    src_seq_len = config['max_seq_len']

    src = torch.randint(0, config['vocab_size'], (batch_size, src_seq_len)).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")
    print(f"Source shape: {src.shape}")
    print("Running forward pass...")
    logits = model(src)

    print(f"Output logits shape: {logits.shape}")
    print(f"Logits:\n{logits}")