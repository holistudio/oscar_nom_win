import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class OscarScriptDataset(Dataset):
    """Pre-tokenized movie screenplays paired with binary Oscar-nomination labels.
    Sequences are right-padded with 0 or truncated to `max_length`.
    """

    def __init__(self, tokenized_items, max_length=5000):
        self.max_length = max_length
        logger.info(f"\nProcessing {len(tokenized_items)} pre-tokenized screenplays into Datasets...")
        self.imdb_ids = []
        self.inputs = []
        self.targets = []

        for idx, item in enumerate(tokenized_items):
            imdb_id = item['imdb_id']
            tokens = item['input_ids']
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            self.imdb_ids.append(imdb_id)
            self.inputs.append(torch.tensor(tokens, dtype=torch.long))
            self.targets.append(torch.tensor(item['target'], dtype=torch.long))

            # if (idx + 1) % 100 == 0:
            #     logger.info(f"  Processed {idx + 1}/{len(tokenized_items)} screenplays")

        logger.info("Processing Datasets complete!")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'imdb_id': self.imdb_ids[idx], 'input_ids': self.inputs[idx], 'target': self.targets[idx]}