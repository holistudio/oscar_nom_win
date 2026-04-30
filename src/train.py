import argparse
import json
import pickle
import importlib
from pathlib import Path
import time
import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from datasets import OscarScriptDataset

import wandb

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

def resolve_resume_path(args, models_dir, checkpoint_prefix):
    """Figure out which checkpoint to resume from, if any.

    Three cases:
      - --resume not passed                  -> None (fresh run)
      - --resume with no path                -> auto-find {prefix}_latest.pth
      - --resume /path/to/checkpoint.pth     -> use that
    """
    if not args.resume_requested:
        return None
    if args.resume_path is not None:
        p = Path(args.resume_path)
        if not p.exists():
            raise FileNotFoundError(f"--resume path does not exist: {p}")
        return p
    auto = models_dir / f'{checkpoint_prefix}_latest.pth'
    if not auto.exists():
        raise FileNotFoundError(
            f"--resume passed but no checkpoint found at {auto}. "
            f"Either pass an explicit path or start fresh without --resume."
        )
    return auto


def warn_on_config_drift(saved_cfg, current_cfg, logger):
    """Log a warning if the resumed config disagrees with the saved one in
    ways that would silently change training semantics."""
    fields_to_check = [
        ('training', 'epochs'),
        ('training', 'batch_size'),
        ('training', 'peak_lr'),
        ('training', 'eta_min'),
        ('training', 'warmup_fraction'),
        ('training', 'class_weights'),
        ('training', 'label_smoothing'),
        ('training', 'weight_decay'),
        ('model', 'params'),
    ]
    drift = []
    for section, key in fields_to_check:
        old = saved_cfg.get(section, {}).get(key)
        new = current_cfg.get(section, {}).get(key)
        if old != new:
            drift.append(f"  {section}.{key}: saved={old!r}  current={new!r}")
    if drift:
        logger.warning(
            "Config differs from saved checkpoint config:\n" + "\n".join(drift) +
            "\nProceeding anyway — make sure this is what you want."
        )

def init_wandb(cfg, results_dir, resume_run_id=None):
    """Initialize a W&B run from the wandb section of the config."""
    wandb_cfg = cfg.get('wandb', {})
    if not wandb_cfg.get('enabled', False):
        return None

    training_cfg = cfg['training']

    default_name = f"{training_cfg['sub_dir']}_{training_cfg['checkpoint_prefix']}"

    run = wandb.init(
        project=wandb_cfg.get('project', 'oscar_nom_win'),
        entity=wandb_cfg.get('entity'),
        mode=wandb_cfg.get('mode', 'offline'),
        name=wandb_cfg.get('name') or default_name,
        notes=wandb_cfg.get('notes', training_cfg.get('notes', '')),
        tags=wandb_cfg.get('tags', training_cfg.get('tags', [])),
        config=cfg,
        dir=str(results_dir),
        id=resume_run_id,
        resume="allow" if resume_run_id else None,
        job_type="train",
    )
    return run

def main():
    # argument parsing
    parser = argparse.ArgumentParser(description='Oscar nomination prediction model trainer')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file')
    parser.add_argument('--resume', dest='resume_path', nargs='?',
                        const='__AUTO__', default=None,
                        help="Resume training. Default auto-loads "
                             "{models_dir}/{sub_dir}/{prefix}_latest.pth. "
                             "Or pass an explicit checkpoint path.")
    args = parser.parse_args()

    # did the user pass --resume at all?
    args.resume_requested = args.resume_path is not None
    if args.resume_path == '__AUTO__':
        args.resume_path = None  # signal: auto-discover

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

    # resolve resume target
    checkpoint_prefix = training_cfg.get('checkpoint_prefix', 'model')
    resume_path = resolve_resume_path(args, models_dir, checkpoint_prefix)

    # save prefix_config.json only on a fresh run
    if resume_path is None:
        save_config_file = models_dir / f"{checkpoint_prefix}_config.json"
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
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
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

    # gradient clipping
    grad_clip = training_cfg.get('grad_clip', 1.0)

    # training loop logging
    history = {'train_loss': [], 'val_loss': [], 
               'train_acc': [], 'val_acc': [],
               'val_prec': [], 'val_rec': [], 'val_f1': [], 'val_auc': []}
    
    best_val_metric = float('-inf')
    start_epoch = 0  # assume for fresh run
    resume_wandb_id = None

    # resume training from checkpoint if --resume specified
    if resume_path is not None:
        logger.info(f"\n>>> RESUMING from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        # config drift check
        if 'config' in ckpt:
            warn_on_config_drift(ckpt['config'], cfg, logger)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        # 'epoch' should be 0-indexed start of the next epoch
        start_epoch = ckpt['epoch']
        history = ckpt.get('history', history)
        best_val_metric = ckpt.get('best_val_metric', float('inf'))

        # pull saved W&B run ID so we can resume the same chart on the cloud
        resume_wandb_id = ckpt.get('wandb_run_id')

        # sanity: the scheduler's internal step count should equal
        # start_epoch * len(train_dataloader)
        expected_steps = start_epoch * len(train_dataloader)
        actual_steps = scheduler.last_epoch
        if actual_steps != expected_steps:
            logger.warning(
                f"Scheduler step count ({actual_steps}) does not match "
                f"start_epoch * steps_per_epoch ({expected_steps}). "
                f"LR schedule may be slightly off."
            )

        if start_epoch >= num_epochs:
            logger.info(
                f"Saved epoch ({start_epoch}) >= configured epochs ({num_epochs}). "
                f"Nothing to do. Increase 'epochs' in config to train further."
            )
            return

        logger.info(
            f">>> Resuming at epoch {start_epoch + 1}/{num_epochs}, "
            f"best_val_metric so far = {best_val_metric:.4f}\n"
        )
        if resume_wandb_id:
            logger.info(f">>> Resuming W&B run id: {resume_wandb_id}")
        logger.info("")
    else:
        logger.info(f"\nStarting fresh training for {num_epochs} epochs...")
    
    # initialize W&B run
    wandb_run = init_wandb(cfg, results_dir, resume_run_id=resume_wandb_id)
    if wandb_run is not None:
        logger.info(
            f"W&B run initialized: id={wandb_run.id}, "
            f"mode={cfg['wandb'].get('mode', 'offline')}, "
            f"dir={wandb_run.dir}"
        )
        # optionally watch the model (gradient histograms)
        if cfg['wandb'].get('watch_model', False):
            wandb.watch(
                model,
                log='gradients',
                log_freq=cfg['wandb'].get('watch_log_freq', 100),
            )

    best_path = models_dir / f'{checkpoint_prefix}_best.pth'
    latest_path = models_dir / f'{checkpoint_prefix}_latest.pth'

    training_start_time = time.time()
    epoch_times = []

    try:
        # training + validation loop
        for epoch in range(start_epoch, num_epochs):
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
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
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
            all_probs = []
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    targets = batch['target'].to(device)

                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        logits = model(input_ids)
                        loss = criterion(logits, targets)

                    preds = torch.argmax(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                    
                    all_probs.extend(probs.cpu().tolist())
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(targets.cpu().tolist())

                    val_losses.append(loss.item())
            # loss
            avg_val_loss = sum(val_losses) / len(val_losses)

            # accuracy
            val_correct = sum(p == t for p, t in zip(all_preds, all_targets))
            avg_val_acc = val_correct / len(all_targets)

            # classification metrics
            try:
                val_auc = roc_auc_score(all_targets, all_probs)
            except ValueError:
                val_auc = float('nan')
            
            val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, labels=[1], average='binary', zero_division=0
            )

            # log timing
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
            history['val_auc'].append(float(val_auc))
            history['val_prec'].append(float(val_prec))
            history['val_rec'].append(float(val_rec))
            history['val_f1'].append(float(val_f1))

            # F1 score default metric for saving best checkpoint
            # AUC is fallback metric when F1 is collapsed to 0, but is mapped to [-1, 0] range
            # so that when F1 does improve, `if val_metric > best_val_metric:` still works below
            val_metric = val_f1 if val_f1 > 0 else (val_auc - 1.0 if not math.isnan(val_auc) else -1.0)

            # build checkpoint payload once, reuse for latest and best models 
            ckpt_payload = {
                'epoch': epoch + 1, # 1-indexed: epoch that just finished
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_auc': float(val_auc),
                'val_f1': float(val_f1),
                'best_val_metric': best_val_metric,
                'history': history,
                'config': cfg,
            }

            # save latest checkpoint
            torch.save(ckpt_payload, latest_path)

            # save best checkpoint when val loss improves
            if val_metric > best_val_metric:
                best_val_metric = val_metric

                torch.save(ckpt_payload, best_path)
                tail = " - New best! Model saved."
                new_best = True
            else:
                tail = ""
                new_best = False

            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Train Acc: {avg_train_acc*100:.1f}%, "
                f"Val Acc: {avg_val_acc*100:.1f}%, "
                f"Val AUC: {val_auc:.3f}, "
                f"Val P/R/F1: {val_prec:.2f}/{val_rec:.2f}/{val_f1:.2f}"
                f" - Elapsed: {elapsed_str}, Avg/Epoch: {avg_time_str}{tail}"
            )

            if wandb_run is not None:
                wandb.log({
                    "train/loss":      avg_train_loss,
                    "train/acc":       avg_train_acc,
                    "val/loss":        avg_val_loss,
                    "val/acc":         avg_val_acc,
                    "val/auc":         float(val_auc),
                    "val/precision":   float(val_prec),
                    "val/recall":      float(val_rec),
                    "val/f1":          float(val_f1),
                    "lr":              scheduler.get_last_lr()[0],
                    "epoch_time_sec":  epoch_time,
                    "best_val_metric": best_val_metric,
                    "new_best":        int(new_best),
                }, step=epoch + 1)

            # save training history every epoch
            history_filename = f"{checkpoint_prefix}_{training_cfg.get('history_filename', 'training_history.json')}"
            history_file = results_dir / history_filename
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)

        logger.info("\nTraining complete!")

        # W&B leaderboard summary metrics
        if wandb_run is not None and len(history['train_loss']) > 0:
            wandb.run.summary["best_val_f1"]  = max(history['val_f1'])

            valid_aucs = [a for a in history['val_auc'] if not math.isnan(a)]
            if valid_aucs:
                wandb.run.summary["best_val_auc"] = max(valid_aucs)

            wandb.run.summary["best_val_acc"]  = max(history['val_acc'])
            wandb.run.summary["min_val_loss"]  = min(history['val_loss'])

            # final-epoch metrics
            wandb.run.summary["final_train_loss"] = history['train_loss'][-1]
            wandb.run.summary["final_val_loss"]   = history['val_loss'][-1]
            wandb.run.summary["epochs_completed"] = len(history['train_loss'])
            
            # model size
            wandb.run.summary["total_params"]     = total_params
            wandb.run.summary["trainable_params"] = trainable_params
    finally:
        # flush W&B even if exiting/crashing mid-training
        if wandb_run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()