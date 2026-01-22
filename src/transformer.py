import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class OscarNomTransformer(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc_d_model = config['enc_d_model']
        self.enc_nhead = config['enc_nhead']
        self.enc_dim_ff = config['enc_dim_ff']
        
        self.agg_d_model = config['agg_d_model']
        self.agg_nhead = config['agg_nhead']
        self.agg_dim_ff = config['agg_dim_ff']
        
        self.chunk_size = config['chunk_size']

        self.token_emb = nn.Embedding(config['vocab_size'], config['enc_d_model'])
        
        self.enc_pos_enc = self._positional_encoder(config['chunk_size'], config['enc_d_model'])
        self.agg_pos_enc = self._positional_encoder(config['max_seq_len'] // config['chunk_size'] + 1, config['agg_d_model'])
        
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['enc_d_model'],
            nhead=config['enc_nhead'],
            dim_feedforward=config['enc_dim_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['enc_num_layers'])

        if config['enc_d_model'] != config['agg_d_model']:
            self.chunk_proj = nn.Linear(config['enc_d_model'], config['agg_d_model'])
        else:
            self.chunk_proj = nn.Identity()
            
        # aggregator
        aggregator_layer = nn.TransformerEncoderLayer(
            d_model=config['agg_d_model'],
            nhead=config['agg_nhead'],
            dim_feedforward=config['agg_dim_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.aggregator = nn.TransformerEncoder(aggregator_layer, num_layers=config['agg_num_layers'])

        self.dropout= nn.Dropout(config['dropout'])

        self.classification_head = nn.Linear(config['agg_d_model'], 2)
        pass

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
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            src = F.pad(src, (0, pad_len), value = 0) # don't pad on left, pad on right with pad_len zeros
            seq_len = src.shape[1] # now reference updated/padded sequence length

        num_chunks = seq_len // self.chunk_size

        # Then reshape to (batch_size * num_chunks, chunk_size)
        src = src.view((batch_size, num_chunks, self.chunk_size))
        src = src.view((batch_size * num_chunks, self.chunk_size))

        # 2. Embed tokens, add positional encoding
        # Shape: (batch_size * num_chunks, chunk_size, enc_d_model)
        src_emb = self.token_emb(src) * math.sqrt(self.enc_d_model)
        src_emb += self.enc_pos_enc[:, :self.chunk_size].to(src_emb.device)
        src_emb = self.dropout(src_emb)
        
        # 3. Encode all chunks (can process in parallel or loop)
        # Shape after encoder: (batch_size * num_chunks, chunk_size, enc_d_model)
        enc_chunks = self.encoder(src_emb)
        
        # 4. Pool each chunk to single vector (mean pool over token dimension)
        # Shape: (batch_size * num_chunks, enc_d_model)
        chunk_embs = enc_chunks.mean(dim=1)

        # 5. Reshape back to (batch_size, num_chunks, enc_d_model)
        chunk_embs = chunk_embs.view((batch_size, num_chunks, self.enc_d_model))
        
        # 6. Project if needed, add chunk positional encoding
        # Shape: (batch_size, num_chunks, dec_d_model)
        
        # 7. Run through aggregator
        # Shape: (batch_size, num_chunks, dec_d_model)
        
        # 8. Pool to single vector (mean pool over chunk dimension)
        # Shape: (batch_size, dec_d_model)
        
        # 9. Classification head
        # Shape: (batch_size, 2)
        
        return logits
    def forward(self, src):
        src_seq_len = src.shape[1]

        src_emb = self.token_emb(src) * math.sqrt(self.enc_d_model)
        src_emb += self.enc_pos_enc[:, :src_seq_len, :].to(src_emb.device)
        src_emb = self.dropout(src_emb)

        memory = self.encoder(src_emb)

        decoder_out = self.decoder(memory)

        logits = self.classification_head(decoder_out)

        logits = logits[:, -1, :]

        return logits
    

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = {
        'chunk_size': 1024,
        'vocab_size': 50257,
        'enc_d_model': 256,
        'enc_nhead': 8,
        'enc_dim_ff': 1024,
        'enc_num_layers': 4,
        
        'agg_d_model': 256,
        'agg_nhead': 8,
        'agg_dim_ff': 1024,
        'agg_num_layers': 4,

        'max_seq_len': 5000,

        'dropout': 0.1
    }

    model = OscarNomTransformer(config).to(device)

    batch_size = 4
    src_seq_len = 512
    tgt_seq_len = 128

    src = torch.randint(0, config['vocab_size'], (batch_size, src_seq_len)).to(device)
    tgt = torch.randint(0, config['vocab_size'], (batch_size, tgt_seq_len)).to(device)

    logits = model(src, tgt)

    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
