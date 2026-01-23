import pandas as pd
import pickle
import os
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

train_df = pd.read_parquet('../data/processed/train_clean.parquet')
val_df = pd.read_parquet('../data/processed/val_clean.parquet')
test_df = pd.read_parquet('../data/processed/test_clean.parquet')

output_folder = './token_data/'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each dataset
datasets = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}

for dataset_name, df in datasets.items():
    print(f"Processing {dataset_name} dataset...")

    # Create list of items
    items = []
    for idx, row in df.iterrows():
        item = {
            'input_ids': tokenizer.encode(row['script_clean']),
            'target': int(row['nominated'])
        }
        items.append(item)

    # Save to pickle file
    output_path = os.path.join(output_folder, f'{dataset_name}_tokenized.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(items, f)

    print(f"Saved {len(items)} items to {output_path}")

print("All datasets processed successfully!")