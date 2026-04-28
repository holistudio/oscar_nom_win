import torch
import torch.nn as nn

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
        self.enc_pos_emb = nn.Embedding(cfg['chunk_size'], cfg['enc_d_model'])
        self.enc_drop_emb = nn.Dropout(cfg['dropout'])
        # TODO: end-of-chunk learnable token

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

        self.max_chunks = cfg['max_seq_len'] // cfg['chunk_size'] + 1
        self.agg_proj = nn.Linear(cfg['enc_d_model'], cfg['agg_d_model']) # TODO: use nn.Identity if enc and agg d_models are same
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

        self.classification_head = nn.Linear(cfg['agg_d_model'], 2)

        enc_causal_mask = torch.triu(
            torch.ones(self.chunk_size, self.chunk_size, dtype=torch.bool), # TODO: needs to change after adding end-of-chunk CLS token
            diagonal=1
        )
        agg_causal_mask = torch.triu(
            torch.ones(self.max_chunks, self.max_chunks, dtype=torch.bool), 
            diagonal=1
        )

        self.register_buffer('enc_causal_mask', enc_causal_mask, persistent=False)
        self.register_buffer('agg_causal_mask', agg_causal_mask, persistent=False)
    
    def forward(self, src):
        batch_size, seq_len = src.shape

        # TODO: padding with end-of-chunk CLS token

        num_chunks = seq_len // self.chunk_size

        # TODO: reshape src for parallel processing

        token_embeds = self.tok_emb(src)
        enc_pos_embeds = self.enc_pos_emb(
            torch.arange(seq_len, device=src.device)
        )
        x = token_embeds + enc_pos_embeds
        x = self.enc_drop_emb(x)

        enc_mask = self.enc_causal_mask[:self.chunk_size, :self.chunk_size]
        x = self.enc_trf_blocks(x, mask=enc_mask, is_causal=True)

        x = x[:, -1, :]

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

