import argparse
import json
import pickle
import importlib
from pathlib import Path
import time
import logging

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

    # save to a subdirectory specified in config
    models_dir = Path(training_cfg.get('models_dir', '../models')) / training_cfg['sub_dir']
    results_dir = Path(training_cfg.get('results_dir', '../results')) / training_cfg['sub_dir']
    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    # save terminal outputs to a log file in the results folder
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(results_dir / f"{training_cfg['checkpoint_prefix']}_training.log")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s • %(name)s: \n%(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # save to prefix_config.json model folder
    save_config_file = models_dir / f"{training_cfg['checkpoint_prefix']}_config.json"
    with open(save_config_file, 'w') as f:
        json.dump(cfg, f, indent=2)

    # seeds for reproducibility
    seed = training_cfg.get('seed', 1337)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    # CUDA device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}\n')

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
    logger.info(f"\nTotal model parameters: {total_params:,}")
    logger.info(f"Trainable model parameters: {trainable_params:,}")
    logger.info(f"Frozen model parameters: {frozen_params:,}")

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

    # TODO: load latest model weights and training/optimizer steps/learning rate schedule

    # gradient clipping
    grad_clip = training_cfg.get('grad_clip', 1.0)

    # save model and results to appropriate directories
    checkpoint_prefix = training_cfg.get('checkpoint_prefix', 'model')

    # training loop logging
    history = {'train_loss': [], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_val_loss = float('inf')
    best_path = models_dir / f'{checkpoint_prefix}_best.pth'

    logger.info(f"\nStarting training for {num_epochs} epochs...")
    training_start_time = time.time()
    epoch_times = []

    # training + validation loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # training step
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += preds.shape[0]

            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = train_correct / train_total

        # validation step
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, targets)
            
            preds = torch.argmax(logits, dim=-1)
            val_correct += (preds == targets).sum().item()
            val_total += preds.shape[0]

            val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = val_correct / val_total

        # log and checkpoint
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        avg_time_str = time.strftime("%H:%M:%S", time.gmtime(avg_epoch_time))

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # save checkpoint
            checkpoint_path = models_dir / f'{checkpoint_prefix}_ep{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': cfg
            }, checkpoint_path)

            # save/overwrite "best_model" file
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': cfg
            }, best_path)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {avg_train_acc*100:.1f}%, "
                  f"Val Acc: {avg_val_acc*100:.1f}% - Elapsed: {elapsed_str}, "
                  f"Avg/Epoch: {avg_time_str} - New best! Model saved.")
        else:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {avg_train_acc*100:.1f}%, "
                  f"Val Acc: {avg_val_acc*100:.1f}% - Elapsed: {elapsed_str}, "
                  f"Avg/Epoch: {avg_time_str}")
    
    # save training history
    history_filename = f"{checkpoint_prefix}_{training_cfg.get('history_filename', 'training_history.json')}"
    history_file = results_dir / history_filename
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("\nTraining complete!")

if __name__ == '__main__':
    main()