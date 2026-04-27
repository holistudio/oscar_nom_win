import argparse
import json
import pickle

import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import OscarScriptDataset

def build_dataloaders(train_dataset, val_dataset, training_cfg, data_cfg, generator):
    bs = training_cfg['batch_size']
    subsample = data_cfg.get('subsample')
    if subsample:
        train_sampler = RandomSampler(
            train_dataset,
            replacement=False,
            num_samples=subsample['train_samples'],
            generator=generator,
        )
        val_sampler = RandomSampler(
            val_dataset,
            replacement=False,
            num_samples=subsample['val_samples'],
            generator=generator,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=bs,
                                  sampler=train_sampler, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                sampler=val_sampler, num_workers=0)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=bs,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                    shuffle=True)
    return train_dataloader, val_dataloader

def main():
    # argument parsing
    parser = argparse.ArgumentParser(description='Oscar nomination prediction model trainer')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    data_cfg = cfg['data']
    model_cfg = cfg['model']
    training_cfg = cfg['training']

    # seeds for reproducibility
    seed = training_cfg.get('seed', 1337)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    # CUDA device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    with open(data_cfg['train_path'], 'rb') as f:
        train_items = pickle.load(f)
    with open(data_cfg['val_path'], 'rb') as f:
        val_items = pickle.load(f)

    max_seq_len = model_cfg['params']['max_seq_len']
    
    train_dataset = OscarScriptDataset(train_items, max_length=max_seq_len)
    val_dataset = OscarScriptDataset(val_items, max_length=max_seq_len)

    train_dataloader, val_dataloader = build_dataloaders(train_dataset, val_dataset,
                                                         training_cfg, data_cfg, generator)
