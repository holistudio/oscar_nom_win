"""
Training script for Oscar nomination prediction using transformer models.

This script trains a hierarchical transformer model to predict whether a movie
will receive an Oscar nomination based on its script. The model uses a chunked
approach to handle long sequences, with separate encoder and aggregator transformers.

The training pipeline includes:
- Data loading from pre-tokenized pickle files
- Model initialization with configurable architecture
- AdamW optimization with weight decay
- Cosine annealing learning rate schedule with linear warmup
- Gradient clipping for training stability
- Model checkpointing based on validation loss
"""

import pickle
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer import OscarNomTransformer


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
    Main training function for the Oscar nomination prediction model.

    This function orchestrates the entire training pipeline:
    1. Loads pre-tokenized training and validation data
    2. Initializes the hierarchical transformer model
    3. Sets up optimizer with AdamW and learning rate scheduling
    4. Trains the model with gradient clipping and validation
    5. Saves best model checkpoints and training history

    The training uses a cosine annealing schedule with linear warmup for
    the learning rate, and implements early stopping based on validation loss.
    """

    # ============================================================================
    # Argument Parsing
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='Train Oscar nomination prediction transformer model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    args = parser.parse_args()

    # ============================================================================
    # Reproducibility
    # ============================================================================
    # Set random seeds for reproducibility across runs
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)  # For multi-GPU setups

    # ============================================================================
    # Configuration
    # ============================================================================
    # Model and training hyperparameters
    config = {
        # Chunking parameters
        'chunk_size': 512,              # Size of each chunk for hierarchical processing
        'vocab_size': 50257,             # Vocabulary size (GPT-2 tokenizer)

        # Encoder transformer (processes individual chunks)
        'enc_d_model': 128,              # Embedding dimension for encoder
        'enc_nhead': 4,                  # Number of attention heads in encoder
        'enc_dim_ff': 512,               # Feedforward dimension in encoder
        'enc_num_layers': 2,             # Number of encoder transformer layers

        # Aggregator transformer (combines chunk representations)
        'agg_d_model': 128,              # Embedding dimension for aggregator
        'agg_nhead': 4,                  # Number of attention heads in aggregator
        'agg_dim_ff': 512,               # Feedforward dimension in aggregator
        'agg_num_layers': 2,             # Number of aggregator transformer layers

        # Sequence parameters
        'max_seq_len': 106578,           # Maximum sequence length (full script)

        # Regularization
        'dropout': 0.3,                  # Dropout probability for all layers

        # Training hyperparameters
        'batch_size': 2,                 # Number of samples per batch
        'peak_lr': 1e-4,                 # Peak learning rate (reached after warmup)
        'weight_decay': 0.05             # L2 regularization coefficient for AdamW
    }

    # ============================================================================
    # Data Loading
    # ============================================================================
    # Load pre-tokenized training data
    # Data is expected to be a list of dicts with 'input_ids' and 'target' keys
    print("Loading training data from ./token_data/train_tokenized.pkl...")
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
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================================
    # Dataset and DataLoader Creation
    # ============================================================================
    # Wrap tokenized data in PyTorch Dataset objects
    # Handles padding/truncation to max_seq_len
    print("\nCreating PyTorch datasets...")
    train_dataset = OscarScriptDataset(train_tokenized_items, max_length=config['max_seq_len'])
    val_dataset = OscarScriptDataset(val_tokenized_items, max_length=config['max_seq_len'])

    # Create dataloaders for batch processing
    print("\nCreating DataLoaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,                    # Shuffle training data for better generalization
        num_workers=0                    # Single-threaded loading (set >0 for parallel)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,                   # No shuffling needed for validation
        num_workers=0
    )

    # ============================================================================
    # Model Initialization
    # ============================================================================
    print("\nInitializing model...")
    model = OscarNomTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============================================================================
    # Loss Function and Optimizer
    # ============================================================================
    # Use CrossEntropyLoss for binary classification (outputs 2 logits)
    criterion = nn.CrossEntropyLoss()

    # AdamW optimizer with decoupled weight decay regularization
    # Initialized with peak_lr (will be modulated by scheduler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['peak_lr'],
        weight_decay=config['weight_decay']
    )

    # ============================================================================
    # Training Configuration
    # ============================================================================
    num_epochs = args.epochs

    # ============================================================================
    # Learning Rate Scheduler
    # ============================================================================
    # Calculate total training steps (batches across all epochs)
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps)  # 10% of training for warmup
    cosine_steps = total_steps - warmup_steps

    # Phase 1: Linear warmup from near-zero to peak learning rate
    # Helps stabilize training in early iterations
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / config['peak_lr'],  # Start from 1e-6
        end_factor=1.0,                          # End at peak_lr
        total_iters=warmup_steps
    )

    # Phase 2: Cosine annealing from peak_lr down to 1e-6
    # Smooth decay helps fine-tune the model in later stages
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,      # Number of steps for cosine decay
        eta_min=1e-6              # Minimum learning rate at end
    )

    # Chain the two schedulers: warmup first, then cosine annealing
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]  # Switch to cosine after warmup_steps
    )

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps} (10%)")
    print(f"Cosine annealing steps: {cosine_steps}")
    print(f"LR schedule: {1e-6:.2e} (warmup start) -> {config['peak_lr']:.2e} (warmup end/peak) -> {1e-6:.2e} (final)")

    # ============================================================================
    # Training Setup
    # ============================================================================
    # Data structure to store training and validation losses per epoch
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Track best validation loss for model checkpointing
    best_val_loss = float('inf')

    # Create directory for saving model checkpoints
    models_dir = Path('../models')
    models_dir.mkdir(exist_ok=True)

    # ============================================================================
    # Main Training Loop
    # ============================================================================
    print(f"\nStarting training for {num_epochs} epochs...")

    # Record start time for training
    training_start_time = time.time()
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # ========================================================================
        # Training Phase
        # ========================================================================
        model.train()  # Set model to training mode (enables dropout, etc.)
        train_losses = []

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch data to device (GPU/CPU)
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            optimizer.zero_grad()             # Clear gradients from previous step
            logits = model(input_ids)         # Get model predictions (2 logits per sample)

            # Compute cross-entropy loss
            loss = criterion(logits, targets)

            # Backward pass and optimization
            loss.backward()                   # Compute gradients via backpropagation

            # Clip gradients to prevent exploding gradients
            # Helps stabilize training, especially with long sequences
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()                  # Update model parameters
            scheduler.step()                  # Update learning rate

            # Track loss for this batch
            train_losses.append(loss.item())

        # Compute average training loss across all batches in this epoch
        avg_train_loss = sum(train_losses) / len(train_losses)

        # ========================================================================
        # Validation Phase
        # ========================================================================
        model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        val_losses = []

        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch data to device
                input_ids = batch['input_ids'].to(device)
                targets = batch['target'].to(device)

                # Forward pass only (no backward pass needed)
                logits = model(input_ids)

                # Compute loss
                loss = criterion(logits, targets)
                val_losses.append(loss.item())

        # Compute average validation loss across all validation batches
        avg_val_loss = sum(val_losses) / len(val_losses)

        # ========================================================================
        # Logging and Checkpointing
        # ========================================================================
        # Calculate timing information
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        # Format timing strings
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        avg_time_str = time.strftime("%H:%M:%S", time.gmtime(avg_epoch_time))

        # Store losses for this epoch in history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Save model checkpoint if validation loss improved (model checkpointing)
        # This implements early stopping: we keep the best model seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = models_dir / 'transformer_best.pth'

            # Save complete training state for potential resumption
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),          # Model weights
                'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                'scheduler_state_dict': scheduler.state_dict(),  # LR scheduler state
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
    # Export training and validation losses to JSON for later analysis/plotting
    print("\nSaving training history...")
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)

    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to {history_file}")
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
