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
        enc_causal_mask = torch.triu(
            torch.ones(self.chunk_size, self.chunk_size, dtype=torch.bool), # TODO: needs to change after adding end-of-chunk CLS token
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
    
    def forward(self, src):
        batch_size, seq_len = src.shape

        # padding if screenplay has extra tokens outside full chunk windows
        # TODO:  with end-of-chunk CLS token
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            src = F.pad(src, (0,pad_len), value=0)
            seq_len = src.shape[1]

        num_chunks = seq_len // self.chunk_size

        # reshape src so that each row is a chunk
        src = src.view(batch_size*num_chunks, self.chunk_size)

        token_embeds = self.tok_emb(src)
        enc_pos_embeds = self.enc_pos_emb(
            torch.arange(seq_len, device=src.device)
        )
        x = token_embeds + enc_pos_embeds
        x = self.enc_drop_emb(x)

        enc_mask = self.enc_causal_mask[:self.chunk_size, :self.chunk_size]
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