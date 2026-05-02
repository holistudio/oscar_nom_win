import logging
import json
from pathlib import Path
import numpy as np

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


def emb_collate_fn(batch):
    """
    Pad variable-length chunk sequences to max length in batch.
    Returns:
        embeddings:        (B, C_max, D) float32
        key_padding_mask:  (B, C_max) bool, True where padded (per nn.Transformer convention)
        target:            (B,) long
        imdb_id:           list[str]
    """
    B = len(batch)
    D = batch[0]["embeddings"].shape[1]
    C_max = max(item["num_chunks"] for item in batch)

    embeddings = torch.zeros(B, C_max, D, dtype=torch.float32)
    key_padding_mask = torch.ones(B, C_max, dtype=torch.bool)  # True = pad
    targets = torch.empty(B, dtype=torch.long)
    imdb_ids = []

    for i, item in enumerate(batch):
        c = item["num_chunks"]
        embeddings[i, :c] = item["embeddings"]
        key_padding_mask[i, :c] = False
        targets[i] = item["target"]
        imdb_ids.append(item["imdb_id"])

    return {
        "embeddings":       embeddings,
        "key_padding_mask": key_padding_mask,
        "target":           targets,
        "imdb_id":          imdb_ids,
    }

class OscarEmbeddingDataset(Dataset):
    """
    One item per screenplay. Reads pre-computed GPT-2 chunk embeddings
    written by emb_generate.py.

    Each .npz contains:
        embeddings    : (num_chunks, emb_dim) float16
        valid_lengths : (num_chunks,) int32
        target        : scalar int
        num_tokens    : scalar int
    """
    def __init__(self, embed_dir, split, eager=False):
        logger.info("\nInitializing screenplay embeddings into Datasets...")
        self.embed_dir = Path(embed_dir)
        self.split = split
        with open(self.embed_dir / f"{split}_manifest.json", "r") as f:
            self.manifest = json.load(f)

        self.eager = eager
        self._cache = None
        if eager:
            # ~130MB total across all splits — fine on a 4090 host
            logger.info("Loading screenplay embeddings into Datasets...")
            self._cache = [self._load(i) for i in range(len(self.manifest))]

    def _load(self, idx):
        entry = self.manifest[idx]
        npz = np.load(self.embed_dir / entry["file"])
        return {
            "embeddings":    npz["embeddings"].astype(np.float32),  # (C, D)
            "valid_lengths": npz["valid_lengths"],
            "target":        int(npz["target"]),
            "imdb_id":       entry.get("imdb_id", ""),
            "num_chunks":    int(npz["embeddings"].shape[0]),
        }

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self._cache[idx] if self.eager else self._load(idx)
        return {
            "embeddings": torch.from_numpy(item["embeddings"]),  # (C, D) float32
            "target":     torch.tensor(item["target"], dtype=torch.long),
            "imdb_id":    item["imdb_id"],
            "num_chunks": item["num_chunks"],
        }