import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class OscarNomTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.chunk_size = cfg['chunk_size']

        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['enc_d_model'])

        # classification / CLS-like summary token at end of each chunk
        self.enc_cls = nn.Parameter(torch.zeros(1,1, cfg['enc_d_model']))
        nn.init.normal_(self.enc_cls, mean=0.0, std=0.02)
        self.enc_seq_len = self.chunk_size + 1

        self.enc_pos_emb = nn.Embedding(self.enc_seq_len, cfg['enc_d_model'])
        self.enc_drop_emb = nn.Dropout(cfg['dropout'])
        
        enc_trf_block = nn.TransformerEncoderLayer(
            d_model=cfg['enc_d_model'],
            nhead=cfg['enc_nhead'],
            dim_feedforward=4 * cfg['enc_d_model'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True,
            bias=True
        )

        self.enc_trf_blocks = nn.TransformerEncoder(
            enc_trf_block,
            num_layers=cfg['enc_num_layers'],
            norm=LayerNorm(cfg['enc_d_model'])
        )
        enc_causal_mask = torch.triu(
            torch.ones(self.enc_seq_len, self.enc_seq_len, dtype=torch.bool), 
            diagonal=1
        )
        self.register_buffer('enc_causal_mask', enc_causal_mask, persistent=False)
        
        self.enc_d_model = cfg['enc_d_model']
        
        if cfg['enc_d_model'] != cfg['agg_d_model']:
            self.agg_proj = nn.Linear(cfg['enc_d_model'], cfg['agg_d_model'])
        else:
            self.agg_proj = nn.Identity()
        self.max_chunks = cfg['max_seq_len'] // cfg['chunk_size'] + 1
        self.agg_pos_emb = nn.Embedding(self.max_chunks, cfg['agg_d_model'])
        self.agg_drop_emb = nn.Dropout(cfg['dropout'])

        agg_trf_block = nn.TransformerEncoderLayer(
            d_model=cfg['agg_d_model'],
            nhead=cfg['agg_nhead'],
            dim_feedforward=4 * cfg['agg_d_model'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True,
            bias=True
        )

        self.agg_trf_blocks = nn.TransformerEncoder(
            agg_trf_block,
            num_layers=cfg['agg_num_layers'],
            norm=LayerNorm(cfg['agg_d_model'])
        )
        agg_causal_mask = torch.triu(
            torch.ones(self.max_chunks, self.max_chunks, dtype=torch.bool), 
            diagonal=1
        )
        self.register_buffer('agg_causal_mask', agg_causal_mask, persistent=False)

        self.classification_head = nn.Linear(cfg['agg_d_model'], 2)

        # Initialize weights following GPT best practices
        self.apply(self._init_weights)

        # Apply special scaled initialization for residual projections
        self.enc_num_layers = cfg['enc_num_layers']
        self.agg_num_layers = cfg['agg_num_layers']
        self._init_residual_projections()
    
    def _init_weights(self, module):
        """
        Initialize weights following GPT best practices.
        - Embeddings: N(0, 0.02)
        - Linear layers: N(0, 0.02) for weights, 0 for biases
        - LayerNorm: standard initialization (weight=1, bias=0)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _init_residual_projections(self):
        """
        Apply scaled initialization to residual projection layers.
        Following GPT-2, scale by 1/sqrt(2*num_layers) for stability in deep networks.
        """
        total_layers = self.enc_num_layers + self.agg_num_layers

        # Scale residual projections in encoder layers
        for layer in self.enc_trf_blocks.layers:
            # Scale the second linear layer in the feedforward network (residual projection)
            torch.nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))
            # Scale the output projection in multi-head attention
            torch.nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))

        # Scale residual projections in aggregator layers
        for layer in self.agg_trf_blocks.layers:
            # Scale the second linear layer in the feedforward network (residual projection)
            torch.nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))
            # Scale the output projection in multi-head attention
            torch.nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))

    def forward(self, src):
        batch_size, seq_len = src.shape

        # padding if screenplay has extra tokens outside full chunk windows
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            src = F.pad(src, (0,pad_len), value=0)
            seq_len = src.shape[1]

        num_chunks = seq_len // self.chunk_size

        # reshape src so that each row is a chunk
        src = src.view(batch_size*num_chunks, self.chunk_size)

        token_embeds = self.tok_emb(src)

        # add CLS-like token at end of each chunk
        cls = self.enc_cls.expand(src.shape[0], -1, -1)
        x = torch.cat([token_embeds, cls], dim=1)
        
        enc_pos_embeds = self.enc_pos_emb(
            torch.arange(self.enc_seq_len, device=src.device)
        )
        x = x+enc_pos_embeds
        x = self.enc_drop_emb(x)

        enc_mask = self.enc_causal_mask[:self.enc_seq_len, :self.enc_seq_len]
        x = self.enc_trf_blocks(x, mask=enc_mask, is_causal=True)

        x = x[:, -1, :]
        # reshape back into batches of screenplays
        x = x.view(batch_size, num_chunks, self.enc_d_model)

        x = self.agg_proj(x)
        agg_pos_embeds = self.agg_pos_emb(
            torch.arange(num_chunks, device=src.device)
        )
        x = x + agg_pos_embeds
        x = self.agg_drop_emb(x)

        agg_mask = self.agg_causal_mask[:num_chunks, :num_chunks]
        x = self.agg_trf_blocks(x, mask=agg_mask, is_causal=True)

        x = x[:, -1, :]

        logits = self.classification_head(x)
        return logits

if __name__ == '__main__':
    torch.manual_seed(1337)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    config = {
        'chunk_size': 1024,
        'vocab_size': 50257,
        'enc_d_model': 256,
        'enc_nhead': 8,
        'enc_num_layers': 4,

        'agg_d_model': 256,
        'agg_nhead': 8,
        'agg_num_layers': 4,

        'max_seq_len': 106578,

        'dropout': 0.1,
    }

    model = OscarNomTransformer(config).to(device)

    batch_size = 2
    src_seq_len = 106578
    src = torch.randint(0, config['vocab_size'], (batch_size, src_seq_len)).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Source shape: {src.shape}")
    logits = model(src)
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits:\n{logits}")