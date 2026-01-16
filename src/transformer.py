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
        self.chunk_size = config['chunk_size']
        self.token_emb = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.pos_enc = PositionalEncoder()
        
        # TODO: re-define as separate encoder and decoder pieces
        self.transformer = nn.Transformer(d_model=config['d_model'],
                                          nhead=config['nhead'],
                                          num_encoder_layers=config['num_encoder_layers'],
                                          num_decoder_layers=config['num_decoder_layers'],
                                          dim_feedforward=config['dim_feedforward'],
                                          dropout=config['dropout'],
                                          batch_first=True)
        self.classification_head = nn.Linear(config['d_model'], 2)
        pass
