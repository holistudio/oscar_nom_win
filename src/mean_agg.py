import math

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class OscarNomAgg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
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
            batch_first=True
        )

        self.agg_trf_blocks = nn.TransformerEncoder(
            agg_trf_block,
            num_layers=cfg['agg_num_layers'],
        )

        self.classification_head = nn.Linear(cfg['agg_d_model'], 2)

        # Initialize weights following GPT best practices
        self.apply(self._init_weights)

        # Apply special scaled initialization for residual projections
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
        batch_size, num_chunks, enc_d_model = src.shape

        x = self.agg_proj(src)
        agg_pos_embeds = self.agg_pos_emb(
            torch.arange(num_chunks, device=src.device)
        )
        x = x + agg_pos_embeds
        x = self.agg_drop_emb(x)

        x = self.agg_trf_blocks(x)

        x = x.mean(dim=1)

        logits = self.classification_head(x)
        return logits

if __name__ == '__main__':
    torch.manual_seed(1337)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    config = {
        'chunk_size': 1024,
        'enc_d_model': 768, # match to GPT-2 (124M)

        'agg_d_model': 256,
        'agg_nhead': 8,
        'agg_num_layers': 4,

        'max_seq_len': 106578,

        'dropout': 0.1,
    }

    model = OscarNomAgg(config).to(device)

    batch_size = 2
    src_seq_len = 106578
    src = torch.randint(0, config['enc_d_model'], (batch_size, src_seq_len)).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Source shape: {src.shape}")
    logits = model(src)
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits:\n{logits}")