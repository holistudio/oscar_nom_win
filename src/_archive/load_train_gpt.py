"""
Resume training script for Oscar nomination prediction using transformer models.

This script loads a saved checkpoint and continues training the hierarchical
transformer model for Oscar nomination prediction. It restores model weights,
optimizer state, and learning rate scheduler state to seamlessly resume
from where a previous training run left off.

NOTE ON SCHEDULER BEHAVIOR:
    This script reconstructs the same SequentialLR (warmup + cosine annealing)
    scheduler used in the original training, then loads the saved scheduler
    state_dict. This means the LR curve continues exactly where it left off
    within the ORIGINAL total step budget (num_epochs * steps_per_epoch).

    If you want to EXTEND training beyond the original epoch count (i.e., the
    cosine schedule has already decayed to eta_min), you should instead build
    a fresh scheduler with a new T_max for the extra epochs. That is a
    different use case and would require modifying the scheduler setup below.

Usage:
    python load_train_gpt.py --epochs 3
    python load_train_gpt.py --epochs 3 --checkpoint ../models/gpt_best_ep1.pth
"""

import os
import pickle
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

import tensorflow as tf
from gpt_download import load_gpt2_params_from_tf_ckpt
from gpt import OscarNomGPT


class OscarScriptDataset(Dataset):
    """
    PyTorch Dataset for Oscar nomination prediction from movie scripts.

    This dataset handles pre-tokenized movie scripts and their corresponding
    Oscar nomination labels. It ensures all sequences are padded or truncated
    to a fixed length for batch processing.
    """

    def __init__(self, tokenized_items, max_length=5000):
        """
        Initialize the dataset with pre-tokenized scripts.

        Args:
            tokenized_items: List of dicts containing:
                - 'input_ids': List of token IDs representing the script
                - 'target': Binary label (0=not nominated, 1=nominated)
            max_length: Maximum sequence length for padding/truncation.
                Longer sequences are truncated, shorter ones are padded with zeros.
        """
        self.max_length = max_length

        # Initialize lists to store processed inputs and targets
        print(f"Processing {len(tokenized_items)} pre-tokenized scripts...")
        self.inputs = []
        self.targets = []

        # Process each script in the dataset
        for idx, item in enumerate(tokenized_items):
            # Extract pre-tokenized token IDs
            tokens = item['input_ids']

            # Ensure uniform sequence length through truncation or padding
            if len(tokens) > max_length:
                # Truncate sequences that exceed max_length
                tokens = tokens[:max_length]
            else:
                # Pad shorter sequences with zeros to reach max_length
                # Note: 0 serves as the padding token ID
                tokens = tokens + [0] * (max_length - len(tokens))

            # Convert to PyTorch tensors and store
            self.inputs.append(torch.tensor(tokens, dtype=torch.long))
            self.targets.append(torch.tensor(item['target'], dtype=torch.long))  # Binary: 0 or 1

            # Progress logging every 100 scripts
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(tokenized_items)} scripts")

        print("Processing complete!")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            dict: Contains 'input_ids' (token tensor) and 'target' (label tensor)
        """
        return {
            'input_ids': self.inputs[idx],
            'target': self.targets[idx]
        }


def main():
    """
    Main function for resuming training of the Oscar nomination prediction model.

    Loads a checkpoint and continues training from the saved epoch, restoring
    model weights, optimizer state, and LR scheduler state.
    """

    # ============================================================================
    # Argument Parsing
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='Resume training Oscar nomination prediction transformer model from checkpoint'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='TOTAL number of training epochs (same as original run, default: 3)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../models/gpt_best_ep1.pth',
        help='Path to checkpoint file to resume from (default: ../models/gpt_best_ep1.pth)'
    )
    args = parser.parse_args()

    model_checkpoint = args.checkpoint

    # ============================================================================
    # Reproducibility
    # ============================================================================
    # Set random seeds for reproducibility across runs
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)  # For multi-GPU setups
    g = torch.Generator() 
    g.manual_seed(1337) # For DataLoader random sampling

    # ============================================================================
    # Configuration
    # ============================================================================
    # Model and training hyperparameters — must match the original training run
    config = {
        # Chunking parameters
        "context_length": 1024,              # Size of each chunk for hierarchical processing
        'vocab_size': 50257,             # Vocabulary size (GPT-2 tokenizer)

        # Encoder GPT-2 (processes individual chunks)
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "qkv_bias": True,

        # Aggregator transformer (combines chunk representations)
        'agg_d_model': 128,              # Embedding dimension for aggregator
        'agg_nhead': 4,                  # Number of attention heads in aggregator
        'agg_dim_ff': 256,               # Feedforward dimension in aggregator
        'agg_num_layers': 1,             # Number of aggregator transformer layers

        # Sequence parameters
        'max_seq_len': 106578,           # Maximum sequence length (full script)

        # Regularization
        "drop_rate": 0.1,                  # Dropout probability for all layers

        # Training hyperparameters
        'batch_size': 1,                 # Number of samples per batch
        'peak_lr': 8e-5,                 # Peak learning rate (reached after warmup)
        'weight_decay': 0.08             # L2 regularization coefficient for AdamW
    }

    # ============================================================================
    # Load Checkpoint
    # ============================================================================
    print(f"Loading checkpoint from {model_checkpoint}...")
    checkpoint = torch.load(model_checkpoint, weights_only=False)
    start_epoch = checkpoint['epoch']  # Epoch to resume from (1-indexed in save, used as range start)
    print(f"Checkpoint loaded: was saved after epoch {start_epoch}")
    print(f"  Checkpoint train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Checkpoint val loss:   {checkpoint['val_loss']:.4f}")

    # ============================================================================
    # Data Loading
    # ============================================================================
    # Load pre-tokenized training data
    print("\nLoading training data from ./token_data/train_tokenized.pkl...")
    with open('./token_data/train_tokenized.pkl', 'rb') as f:
        train_tokenized_items = pickle.load(f)
    print(f"Loaded {len(train_tokenized_items)} training samples")

    # Load pre-tokenized validation data
    print("Loading validation data from ./token_data/val_tokenized.pkl...")
    with open('./token_data/val_tokenized.pkl', 'rb') as f:
        val_tokenized_items = pickle.load(f)
    print(f"Loaded {len(val_tokenized_items)} validation samples")

    # ============================================================================
    # Device Setup
    # ============================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================================
    # Dataset and DataLoader Creation
    # ============================================================================
    print("\nCreating PyTorch datasets...")
    train_dataset = OscarScriptDataset(train_tokenized_items, max_length=config['max_seq_len'])
    val_dataset = OscarScriptDataset(val_tokenized_items, max_length=config['max_seq_len'])

    # Random sampling subset
    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=185)
    val_sampler = RandomSampler(val_dataset, replacement=False, num_samples=122)

    # Create dataloaders for batch processing
    print("\nCreating DataLoaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=0
    )

    # ============================================================================
    # Model Initialization
    # ============================================================================
    # We still need to construct the model architecture (including loading GPT-2
    # params to get the right shapes), then overwrite with checkpoint weights.
    print("\nLoading GPT-2 pretrained weights (for architecture construction)...")
    gpt_size="124M"
    gpts_dir="gpt2"

    gpt_dir = os.path.join(gpts_dir, gpt_size)
    tf_ckpt_path = tf.train.latest_checkpoint(gpt_dir)
    settings = json.load(open(os.path.join(gpt_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    print("\nInitializing model architecture...")
    model = OscarNomGPT(config, params).to(device)

    # Now overwrite all weights with the checkpoint's trained weights
    print("Loading trained weights from checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")

    # ============================================================================
    # Loss Function and Optimizer
    # ============================================================================
    class_weights = torch.tensor([1.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['peak_lr'],
        weight_decay=config['weight_decay']
    )

    # Restore optimizer state (momentum buffers, adaptive learning rates, etc.)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ============================================================================
    # Training Configuration
    # ============================================================================
    num_epochs = args.epochs  # Total epochs (same as original training plan)

    if start_epoch >= num_epochs:
        print(f"\nCheckpoint is from epoch {start_epoch}, but total epochs is {num_epochs}.")
        print("Nothing left to train. Increase --epochs if you want to extend training.")
        return

    # ============================================================================
    # Learning Rate Scheduler
    # ============================================================================
    # Reconstruct the SAME scheduler structure as the original training run,
    # then load the saved state_dict so the internal step counter is correct.
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps)
    cosine_steps = total_steps - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / config['peak_lr'],
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Restore scheduler state (internal step counters, last_epoch, etc.)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"\nResuming from epoch {start_epoch + 1} / {num_epochs}")
    print(f"Total training steps (original plan): {total_steps}")
    print(f"Warmup steps: {warmup_steps} (10%)")
    print(f"Cosine annealing steps: {cosine_steps}")

    # ============================================================================
    # Training Setup
    # ============================================================================
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Initialize best_val_loss from checkpoint so we only save if we improve
    best_val_loss = checkpoint['val_loss']
    print(f"Best val loss from checkpoint: {best_val_loss:.4f}")

    models_dir = Path('../models')
    models_dir.mkdir(exist_ok=True)

    # ============================================================================
    # Main Training Loop (resumed)
    # ============================================================================
    print(f"\nResuming training from epoch {start_epoch + 1} to {num_epochs}...")

    training_start_time = time.time()
    epoch_times = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        # ========================================================================
        # Training Phase
        # ========================================================================
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # ========================================================================
        # Validation Phase
        # ========================================================================
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['target'].to(device)

                logits = model(input_ids)

                loss = criterion(logits, targets)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        # ========================================================================
        # Logging and Checkpointing
        # ========================================================================
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        avg_time_str = time.strftime("%H:%M:%S", time.gmtime(avg_epoch_time))

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = models_dir / f'gpt_best_ep{epoch+1}.pth'

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} - "
                  f"Elapsed: {elapsed_str}, Avg/Epoch: {avg_time_str} - New best! Model saved.")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} - "
                  f"Elapsed: {elapsed_str}, Avg/Epoch: {avg_time_str}")

    # ============================================================================
    # Save Training History
    # ============================================================================
    print("\nSaving training history...")
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)

    history_file = output_dir / 'resumed_training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to {history_file}")
    print("\nResumed training complete!")

if __name__ == '__main__':
    main()