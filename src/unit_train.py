import pandas as pd

import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader

from transformer import OscarNomTransformer


class OscarScriptDataset(Dataset):
    """Dataset for Oscar nomination prediction from movie scripts."""

    def __init__(self, df, tokenizer, max_length=5000):
        """
        Args:
            df: DataFrame with 'script_clean' and 'nominated' columns
            tokenizer: tiktoken tokenizer instance
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize all scripts and store targets
        print(f"Tokenizing {len(df)} scripts...")
        self.inputs = []
        self.targets = []

        for idx, row in df.iterrows():
            # Tokenize script text
            tokens = tokenizer.encode(row['script_clean'])

            # Truncate or pad to max_length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                # Pad with 0s (or use a specific pad token if needed)
                tokens = tokens + [0] * (max_length - len(tokens))

            # Store tokenized input and target
            self.inputs.append(torch.tensor(tokens, dtype=torch.long))
            self.targets.append(int(row['nominated']))  # 0 or 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} scripts")

        print(f"Tokenization complete!")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }


def main():
    # Load training data
    print("Loading data from data/processed/train_clean.parquet...")
    df = pd.read_parquet('data/processed/train_clean.parquet')
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Initialize tokenizer
    print("\nInitializing GPT-2 tokenizer (tiktoken)...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    print("\nCreating PyTorch dataset...")
    dataset = OscarScriptDataset(df, tokenizer, max_length=106578)

    print("\nCreating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # Load random sample from DataLoader
    print("\nLoading random sample from DataLoader...")
    first_batch = next(iter(dataloader))
    src = first_batch['input_ids'].to(device)  # [batch_size, seq_len]
    target = first_batch['target'].to(device)  # [batch_size]
    
    print(f"\nFirst sample:")
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
    print(f"\nModel Output:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits: {logits}")
    print(f"  Raw values: {logits[0].cpu().numpy()}")

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    print(f"\nProbabilities:")
    print(f"  Class 0 (Not nominated/won): {probs[0, 0].item():.4f}")
    print(f"  Class 1 (Nominated/won): {probs[0, 1].item():.4f}")

    # Get prediction
    prediction = torch.argmax(logits, dim=-1)
    print(f"\nPrediction: {prediction.item()}")

if __name__ == '__main__':
    main()
