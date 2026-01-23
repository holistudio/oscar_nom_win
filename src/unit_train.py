import pickle

import torch
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
    # Load pre-tokenized training data
    print("Loading data from ./token_data/train_tokenized.pkl...")
    with open('./token_data/train_tokenized.pkl', 'rb') as f:
        tokenized_items = pickle.load(f)
    print(f"Loaded {len(tokenized_items)} samples")

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    print("\nCreating PyTorch dataset...")
    dataset = OscarScriptDataset(tokenized_items, max_length=106578)

    print("\nCreating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    # Load random sample from DataLoader
    print("\nLoading random sample from DataLoader...")
    first_batch = next(iter(dataloader))
    src = first_batch['input_ids'].to(device)  # [batch_size, seq_len]
    target = first_batch['target'].to(device)  # [batch_size]
    
    print("\nFirst sample:")
    print(f"  Input shape: {src.shape}")
    print(f"  Input (first 10 tokens): {src[0, :10]}")
    print(f"  Target: {target.item()}")
    print(f"  Target label: {'Nominated' if target.item() == 1 else 'Not nominated'}")

    
    # Model configuration
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

        'max_seq_len': 106578,

        'dropout': 0.1
    }

    # Initialize model
    print("\nInitializing OscarNomTransformer...")
    model = OscarNomTransformer(config).to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run inference
    print("\nRunning forward pass...")
    with torch.no_grad():
        logits = model(src)

    # Print results
    print("\nModel Output:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits: {logits}")
    print(f"  Raw values: {logits[0].cpu().numpy()}")

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    print("\nProbabilities:")
    print(f"  Class 0 (Not nominated/won): {probs[0, 0].item():.4f}")
    print(f"  Class 1 (Nominated/won): {probs[0, 1].item():.4f}")

    # Get prediction
    prediction = torch.argmax(logits, dim=-1)
    print(f"\nPrediction: {prediction.item()}")

if __name__ == '__main__':
    main()
