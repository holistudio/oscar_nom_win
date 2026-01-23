import pickle
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer import OscarNomTransformer


class OscarScriptDataset(Dataset):
    """Dataset for Oscar nomination prediction from movie scripts."""

    def __init__(self, tokenized_items, max_length=5000):
        """
        Args:
            tokenized_items: List of dicts with 'input_ids' and 'target' keys
            max_length: Maximum sequence length for padding/truncation
        """
        self.max_length = max_length

        # Process pre-tokenized inputs
        print(f"Processing {len(tokenized_items)} pre-tokenized scripts...")
        self.inputs = []
        self.targets = []

        for idx, item in enumerate(tokenized_items):
            # Get pre-tokenized input_ids
            tokens = item['input_ids']

            # Truncate or pad to max_length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                # Pad with 0s (or use a specific pad token if needed)
                tokens = tokens + [0] * (max_length - len(tokens))

            # Store tokenized input and target
            self.inputs.append(torch.tensor(tokens, dtype=torch.long))
            self.targets.append(torch.tensor(item['target'], dtype=torch.long))  # 0 or 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(tokenized_items)} scripts")

        print("Processing complete!")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'target': self.targets[idx]
        }


def main():

    # Model and training configuration
    config = {
        'chunk_size': 512,
        'vocab_size': 50257,
        'enc_d_model': 128,
        'enc_nhead': 4,
        'enc_dim_ff': 512,
        'enc_num_layers': 2,
        
        'agg_d_model': 128,
        'agg_nhead': 4,
        'agg_dim_ff': 512,
        'agg_num_layers': 2,

        'max_seq_len': 106578,

        'dropout': 0.3,

        'batch_size': 2,
        'peak_lr': 1e-4,
        'weight_decay': 0.05
    }

    # Load pre-tokenized training data
    print("Loading training data from ./token_data/train_tokenized.pkl...")
    with open('./token_data/train_tokenized.pkl', 'rb') as f:
        train_tokenized_items = pickle.load(f)
    print(f"Loaded {len(train_tokenized_items)} training samples")

    # Load pre-tokenized validation data
    print("Loading validation data from ./token_data/val_tokenized.pkl...")
    with open('./token_data/val_tokenized.pkl', 'rb') as f:
        val_tokenized_items = pickle.load(f)
    print(f"Loaded {len(val_tokenized_items)} validation samples")

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = OscarScriptDataset(train_tokenized_items, max_length=config['max_seq_len'])
    val_dataset = OscarScriptDataset(val_tokenized_items, max_length=config['max_seq_len'])

    # Create dataloaders
    print("\nCreating DataLoaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    print("\nInitializing model...")
    model = OscarNomTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['peak_lr'], weight_decay=config['weight_decay'])

    # Training configuration
    num_epochs = 100

    # Calculate total steps for learning rate scheduling
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps)
    cosine_steps = total_steps - warmup_steps

    # Create learning rate scheduler with warmup and cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / config['peak_lr'],  # Start from very small lr
        end_factor=1.0,  # End at peak lr
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

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps} (10%)")
    print(f"Cosine annealing steps: {cosine_steps}")
    print(f"LR schedule: {1e-6:.2e} (warmup start) -> {config['peak_lr']:.2e} (warmup end/peak) -> {1e-6:.2e} (final)")

    # Data structure to store losses
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Initialize best validation loss for model checkpointing
    best_val_loss = float('inf')

    # Create models directory
    models_dir = Path('../models')
    models_dir.mkdir(exist_ok=True)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_dataloader):
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)

            # Calculate loss
            loss = criterion(logits, targets)

            # Backward pass and optimization
            loss.backward()
            # Gradient clipping at 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        # Calculate average training loss for the epoch
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Get batch data
                input_ids = batch['input_ids'].to(device)
                targets = batch['target'].to(device)

                # Forward pass
                logits = model(input_ids)

                # Calculate loss
                loss = criterion(logits, targets)
                val_losses.append(loss.item())

        # Calculate average validation loss for the epoch
        avg_val_loss = sum(val_losses) / len(val_losses)

        # Store losses in history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Save model checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = models_dir / f'AnyModelClass_best_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} - New best! Model saved.")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save training history
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
