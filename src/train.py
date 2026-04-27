import argparse
import json
import pickle
import importlib

import torch
import torch.nn as nn
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

def import_model_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def build_model(model_cfg, device):
    ModelClass = import_model_class(model_cfg['module'], model_cfg['class_name'])

    model = ModelClass(model_cfg['params']).to(device)
    return model

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
    
    # define model
    model = build_model(model_cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}")
    print(f"Frozen model parameters: {frozen_params:,}")

    # loss criterion
    class_weights = torch.tensor(training_cfg.get('class_weights', [1.0, 1.0]), 
                                 dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=training_cfg.get('label_smoothing', 0.0)
    )

    # optimizer 
    peak_lr = training_cfg.get('peak_lr', 3e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=training_cfg.get('weight_decay', 0.1)
    )

    # learning rate scheduler
    num_epochs = training_cfg['epochs']
    eta_min = training_cfg.get('eta_min', 1e-6)
    warmup_fraction = training_cfg.get('warmup_fraction', 0.1)

    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = max(1, int(warmup_fraction * total_steps))
    cosine_steps = total_steps - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=eta_min/peak_lr,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=eta_min
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # gradient clipping
    grad_clip = training_cfg.get('grad_clip', 1.0)
    
    # TODO: load latest model weights and training/optimizer steps/learning rate schedule