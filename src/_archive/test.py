import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt




model_path = '../models/20260123_100epochs/transformer_best_ep7.pth'

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


class OscarNomTransformer(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc_d_model = config['enc_d_model']
        self.enc_nhead = config['enc_nhead']
        self.enc_dim_ff = config['enc_dim_ff']
        
        self.agg_d_model = config['agg_d_model']
        self.agg_nhead = config['agg_nhead']
        self.agg_dim_ff = config['agg_dim_ff']
        
        self.chunk_size = config['chunk_size']
        self.enc_num_layers = config['enc_num_layers']
        self.agg_num_layers = config['agg_num_layers']

        self.token_emb = nn.Embedding(config['vocab_size'], config['enc_d_model'])
        
        self.enc_pos_enc = self._positional_encoder(config['chunk_size'], config['enc_d_model'])
        self.agg_pos_enc = self._positional_encoder(config['max_seq_len'] // config['chunk_size'] + 1, config['agg_d_model'])
        
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['enc_d_model'],
            nhead=config['enc_nhead'],
            dim_feedforward=config['enc_dim_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['enc_num_layers'])

        if config['enc_d_model'] != config['agg_d_model']:
            self.chunk_proj = nn.Linear(config['enc_d_model'], config['agg_d_model'])
        else:
            self.chunk_proj = nn.Identity()
            
        # aggregator
        aggregator_layer = nn.TransformerEncoderLayer(
            d_model=config['agg_d_model'],
            nhead=config['agg_nhead'],
            dim_feedforward=config['agg_dim_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.aggregator = nn.TransformerEncoder(aggregator_layer, num_layers=config['agg_num_layers'])

        self.dropout= nn.Dropout(config['dropout'])

        self.classification_head = nn.Linear(config['agg_d_model'], 2)

        # Initialize weights following GPT best practices
        self.apply(self._init_weights)

        # Apply special scaled initialization for residual projections
        self._init_residual_projections()

    def _positional_encoder(self, max_seq_len, d_model):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self, module):
        """
        Initialize weights following GPT best practices.
        - Embeddings: N(0, 0.02)
        - Linear layers: N(0, 0.02) for weights, 0 for biases
        - LayerNorm: standard initialization (weight=1, bias=0)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _init_residual_projections(self):
        """
        Apply scaled initialization to residual projection layers.
        Following GPT-2, scale by 1/sqrt(2*num_layers) for stability in deep networks.
        """
        total_layers = self.enc_num_layers + self.agg_num_layers

        # Scale residual projections in encoder layers
        for layer in self.encoder.layers:
            # Scale the second linear layer in the feedforward network (residual projection)
            torch.nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))
            # Scale the output projection in multi-head attention
            torch.nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))

        # Scale residual projections in aggregator layers
        for layer in self.aggregator.layers:
            # Scale the second linear layer in the feedforward network (residual projection)
            torch.nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))
            # Scale the output projection in multi-head attention
            torch.nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * total_layers))

    def forward(self, src):
        batch_size, seq_len = src.shape
        
        # 1. Chunk the input
        # Pad seq_len to be divisible by chunk_size
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            src = F.pad(src, (0, pad_len), value = 0) # don't pad on left, pad on right with pad_len zeros
            seq_len = src.shape[1] # now reference updated/padded sequence length

        num_chunks = seq_len // self.chunk_size

        # Then reshape to (batch_size * num_chunks, chunk_size)
        src = src.view((batch_size, num_chunks, self.chunk_size))
        src = src.view((batch_size * num_chunks, self.chunk_size))

        # 2. Embed tokens, add positional encoding
        # Shape: (batch_size * num_chunks, chunk_size, enc_d_model)
        src_emb = self.token_emb(src) * math.sqrt(self.enc_d_model)
        src_emb += self.enc_pos_enc[:, :self.chunk_size].to(src_emb.device)
        src_emb = self.dropout(src_emb)
        
        # 3. Encode all chunks (can process in parallel or loop)
        # Shape after encoder: (batch_size * num_chunks, chunk_size, enc_d_model)
        enc_chunks = self.encoder(src_emb)
        
        # 4. Pool each chunk to single vector (mean pool over token dimension)
        # Shape: (batch_size * num_chunks, enc_d_model)
        chunk_embs = enc_chunks.mean(dim=1)

        # 5. Reshape back to (batch_size, num_chunks, enc_d_model)
        chunk_embs = chunk_embs.view((batch_size, num_chunks, self.enc_d_model))

        # 6. Project if needed, add chunk positional encoding
        # Shape: (batch_size, num_chunks, agg_d_model)
        chunk_embs = self.chunk_proj(chunk_embs) * math.sqrt(self.agg_d_model)
        chunk_embs += self.agg_pos_enc[:, :num_chunks, :].to(chunk_embs.device)
        chunk_embs = self.dropout(chunk_embs)
        
        # 7. Run through aggregator
        # Shape: (batch_size, num_chunks, agg_d_model)
        agg_out = self.aggregator(chunk_embs)
        
        # 8. Pool to single vector (mean pool over chunk dimension)
        # Shape: (batch_size, agg_d_model)
        agg_out = agg_out.mean(dim=1)
        
        # 9. Classification head
        # Shape: (batch_size, 2)
        logits = self.classification_head(agg_out)
        
        return logits
    


def main():


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
        'batch_size': 1,                 # Number of samples per batch
        'peak_lr': 1e-4,                 # Peak learning rate (reached after warmup)
        'weight_decay': 0.05             # L2 regularization coefficient for AdamW
    }

    # ============================================================================
    # Data Loading
    # ============================================================================
    test_token_path = '../src/token_data/test_tokenized.pkl'
    
    # Load pre-tokenized test data
    # Data is expected to be a list of dicts with 'input_ids' and 'target' keys
    print(f"Loading test data from {test_token_path}...")
    with open(test_token_path, 'rb') as f:
        test_tokenized_items = pickle.load(f)
    print(f"Loaded {len(test_tokenized_items)} test samples")

    # ============================================================================
    # Device Setup
    # ============================================================================
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================================
    # Dataset and DataLoader Creation
    # ============================================================================
    # Wrap tokenized data in PyTorch Dataset object
    # Handles padding/truncation to max_seq_len
    print("\nCreating PyTorch dataset...")
    test_dataset = OscarScriptDataset(test_tokenized_items, max_length=config['max_seq_len'])

    # Create dataloader for batch processing
    print("\nCreating DataLoader...")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,                   # No shuffling needed for test data
        num_workers=0                    # Single-threaded loading (set >0 for parallel)
    )

    # ============================================================================
    # Model Initialization
    # ============================================================================
    print("\nInitializing model...")
    model = OscarNomTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pre-trained weights from checkpoint
    print(f"\nLoading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded successfully from epoch {checkpoint['epoch']}!")
    model.eval()  # Set model to evaluation mode for testing

    # ============================================================================
    # Model Evaluation
    # ============================================================================
    print("\nEvaluating model on test set...")

    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            # Forward pass - get model predictions
            logits = model(input_ids)

            # Get predicted class (0 or 1)
            predictions = torch.argmax(logits, dim=1)

            # Update accuracy metrics
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            # Store predictions and targets for confusion matrix
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Progress update
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_dataloader)} batches")

    # Calculate and display final accuracy
    accuracy = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"{'='*60}")

    # ============================================================================
    # Confusion Matrix
    # ============================================================================
    print("\nGenerating confusion matrix...")

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # Display confusion matrix as text
    print("\nConfusion Matrix:")
    print(f"{'':20} Predicted Not Nominated  Predicted Nominated")
    print(f"{'Actually Not Nominated':20} {cm[0, 0]:>22} {cm[0, 1]:>19}")
    print(f"{'Actually Nominated':20} {cm[1, 0]:>22} {cm[1, 1]:>19}")

    # Visualize confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Not Nominated (0)', 'Nominated (1)']
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Oscar Nomination Predictions')
    plt.tight_layout()

    # Save the figure
    output_path = '../models/20260123_100epochs/confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {output_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()