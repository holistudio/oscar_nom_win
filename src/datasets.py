import torch
from torch.utils.data import Dataset

class OscarScriptDataset(Dataset):
    """Pre-tokenized movie screenplays paired with binary Oscar-nomination labels.
    Sequences are right-padded with 0 or truncated to `max_length`.
    """

    def __init__(self, tokenized_items, max_length=5000):
        self.max_length = max_length
        print(f"\nProcessing {len(tokenized_items)} pre-tokenized screenplays into Datasets...")
        self.inputs = []
        self.targets = []

        for idx, item in enumerate(tokenized_items):
            tokens = item['input_ids']
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            self.inputs.append(torch.tensor(tokens, dtype=torch.long))
            self.targets.append(torch.tensor(item['target'], dtype=torch.long))

            # if (idx + 1) % 100 == 0:
            #     print(f"  Processed {idx + 1}/{len(tokenized_items)} screenplays")

        print("Processing Datasets complete!")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx], 'target': self.targets[idx]}