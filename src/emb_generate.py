"""
Generate GPT-2 (124M) embeddings for movie screenplays.

For each screenplay, we chunk the token sequence into non-overlapping windows
of `context_length` (1024) tokens, run the GPT embedding model in eval/no_grad
to get the 768-d hidden states, and save one .npz file per screenplay.

Output structure:
  ./gpt2_embed/train/<idx>.npz   (one per screenplay)
  ./gpt2_embed/val/<idx>.npz
  ./gpt2_embed/test/<idx>.npz
  ./gpt2_embed/<split>_manifest.json   (index of files + metadata)

Each .npz contains:
  - 'embeddings'    : (num_chunks, 768) float16
                      The last *valid* token's hidden state for each chunk.
                      GPT-2 is causal, so this token has attended to every
                      preceding token in the chunk -- it's the natural
                      "summary" vector for a decoder-only model.
  - 'valid_lengths' : (num_chunks,) int32  -- number of real (non-pad) tokens
                      in each chunk. The last chunk is usually partial; we
                      use this to index the correct last-token position.
  - 'target'        : scalar int           -- the label (0/1)
  - 'num_tokens'    : scalar int           -- original token count

Notes:
  - We use bf16 autocast on the forward pass for ~2x speedup on the 4090.
    Embeddings are stored as float16 to halve disk usage with negligible loss.
  - Chunks are batched within a screenplay for GPU efficiency.
"""

import os
import json
import pickle
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import tensorflow as tf
from gpt_download import load_gpt2_params_from_tf_ckpt
from emb_gpt import GPTEmbedModel


# ----------------------------- config ------------------------------------- #

CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,   # eval mode -> no dropout anyway, but be explicit
    "qkv_bias": True,   # GPT-2 uses bias on QKV
}

MODEL_SIZE = "124M"
MODELS_DIR = "gpt2"

TOKEN_DATA_DIR = Path("./token_data")
EMBED_DIR = Path("./gpt2_embed")

SPLITS = ["train", "val", "test"]

# How many 1024-token chunks to forward through the GPU at once.
# 32 chunks * 1024 tokens * 768 dim * 2 bytes (bf16) = ~50 MB activations base.
# Full attention activations are larger but still fits comfortably in 16GB.
CHUNK_BATCH_SIZE = 32


# ----------------------------- model -------------------------------------- #

def build_model(device):
    """Load GPT-2 124M pretrained weights into the embedding model."""
    model_dir = os.path.join(MODELS_DIR, MODEL_SIZE)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    model = GPTEmbedModel(CONFIG, params=params)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded GPTEmbedModel ({n_params:,} parameters) on {device}")
    return model


# ----------------------------- chunking ----------------------------------- #

def chunk_tokens(token_ids, context_length):
    """
    Split a 1D list/array of token ids into non-overlapping (context_length,)
    chunks. The final chunk is right-padded with 0 if shorter than context_length.

    Returns:
        chunks:        (num_chunks, context_length) int64 tensor
        valid_lengths: (num_chunks,)               int32 array
    """
    tokens = np.asarray(token_ids, dtype=np.int64)
    n = len(tokens)
    num_chunks = max(1, (n + context_length - 1) // context_length)

    chunks = np.zeros((num_chunks, context_length), dtype=np.int64)
    valid_lengths = np.zeros(num_chunks, dtype=np.int32)

    for i in range(num_chunks):
        start = i * context_length
        end = min(start + context_length, n)
        chunks[i, : end - start] = tokens[start:end]
        valid_lengths[i] = end - start

    return torch.from_numpy(chunks), valid_lengths


# ----------------------------- embedding pass ----------------------------- #

@torch.no_grad()
def embed_screenplay(model, token_ids, device, dtype=torch.bfloat16):
    """
    Run GPT-2 over all chunks of a single screenplay.

    For each chunk we extract the hidden state at the *last valid token*
    position (i.e. valid_length - 1). Because GPT-2 is causal, that token
    has attended to every preceding real token in the chunk -- it's the
    natural summary vector.

    Returns:
        embeddings    : (num_chunks, 768) float16 numpy array
        valid_lengths : (num_chunks,) int32 numpy array
    """
    chunks, valid_lengths = chunk_tokens(token_ids, CONFIG["context_length"])
    num_chunks = chunks.shape[0]

    out_buffers = []
    for batch_start in range(0, num_chunks, CHUNK_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + CHUNK_BATCH_SIZE].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            hidden = model(batch)  # (B, 1024, 768)

        vl = torch.from_numpy(
            valid_lengths[batch_start : batch_start + CHUNK_BATCH_SIZE]
        ).to(device)
        last_idx = (vl - 1).clamp(min=0).long()  # (B,)

        # gather the hidden state at last_idx for each row in the batch
        b_idx = torch.arange(hidden.shape[0], device=device)
        last_hidden = hidden[b_idx, last_idx, :]  # (B, 768)

        out_buffers.append(last_hidden.to(torch.float16).cpu().numpy())

    embeddings = np.concatenate(out_buffers, axis=0)
    return embeddings, valid_lengths


# ----------------------------- driver ------------------------------------- #

def process_split(split, model, device):
    in_path = TOKEN_DATA_DIR / f"{split}_tokenized.pkl"
    out_dir = EMBED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing split: {split} ===")
    print(f"Loading {in_path}...")
    with open(in_path, "rb") as f:
        items = pickle.load(f)
    print(f"Loaded {len(items)} screenplays")

    manifest = []
    t0 = time.time()

    for idx, item in enumerate(items):
        token_ids = item["input_ids"]
        target = int(item["target"])

        # Optional: include any metadata you have (imdb_id, title, year)
        meta_extras = {
            k: item[k] for k in ("imdb_id", "title", "year", "title_year")
            if k in item
        }

        embeddings, valid_lengths = embed_screenplay(model, token_ids, device)

        out_path = out_dir / f"{idx:05d}.npz"
        np.savez(
            out_path,
            embeddings=embeddings,
            valid_lengths=valid_lengths,
            target=np.int32(target),
            num_tokens=np.int32(len(token_ids)),
        )

        manifest.append({
            "idx": idx,
            "file": str(out_path.relative_to(EMBED_DIR)),
            "target": target,
            "num_tokens": len(token_ids),
            "num_chunks": int(embeddings.shape[0]),
            **meta_extras,
        })

        if (idx + 1) % 25 == 0 or idx == len(items) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(items) - idx - 1) / rate
            print(
                f"  [{idx+1:4d}/{len(items)}] "
                f"chunks={embeddings.shape[0]:3d}  "
                f"{rate:.2f} scripts/s  ETA {eta/60:.1f} min"
            )

    manifest_path = EMBED_DIR / f"{split}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS, choices=SPLITS,
        help="Which splits to process",
    )
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(device)

    for split in args.splits:
        process_split(split, model, device)

    print("\nDone.")


if __name__ == "__main__":
    main()