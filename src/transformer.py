import math

import torch
import torch.nn as nn

config = {}

# TODO: positional encoder and think carefully where it is placed.
class PositionalEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

class OscarNomTransformer(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc_d_model = config['enc_d_model']
        self.enc_nhead = config['enc_nhead']
        self.enc_dim_ff = config['enc_dim_ff']
        
        self.dec_d_model = config['dec_d_model']
        self.dec_nhead = config['dec_nhead']
        self.dec_dim_ff = config['dec_dim_ff']
        
        self.chunk_size = config['chunk_size']

        self.token_emb = nn.Embedding(config['vocab_size'], config['enc_d_model'])
        self.tgt_emb = nn.Embedding(config['vocab_size'], config['dec_d_model'])
        
        self.enc_pos_enc = self._positional_encoder(config['max_seq_len'], config['enc_d_model'])
        self.dec_pos_enc = self._positional_encoder(config['max_seq_len'], config['dec_d_model'])
        
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['enc_d_model'],
            nhead=config['enc_nhead'],
            dim_feedforward=config['enc_dim_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['enc_num_layers'])

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config['dec_d_model'],
            nhead=config['dec_nhead'],
            dim_feedforward=config['dec_dim_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['dec_num_layers'])

        self.dropout= nn.Dropout(config['dropout'])

        self.classification_head = nn.Linear(config['dec_d_model'], 2)
        pass

    def _positional_encoder(self, max_seq_len, d_model):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]


        src_emb = self.token_emb(src) * math.sqrt(self.enc_d_model)
        src_emb += self.enc_pos_enc[:, :src_seq_len, :].to(src_emb.device)
        src_emb = self.dropout(src_emb)

        tgt_emb = self.tgt_emb(tgt) * math.sqrt(self.dec_d_model)
        tgt_emb += self.dec_pos_enc[:, :tgt_seq_len, :].to(tgt_emb.device)
        tgt_emb = self.dropout(tgt_emb)

        memory = self.encoder(src_emb)

        decoder_out = self.decoder(tgt_emb, memory)

        logits = self.classification_head(decoder_out)

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
        
        'dec_d_model': 256,
        'dec_nhead': 8,
        'dec_dim_ff': 1024,
        'dec_num_layers': 4,

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
