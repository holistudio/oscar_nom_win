import argparse
import json
import importlib
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from datasets import OscarEmbeddingDataset, emb_collate_fn

import wandb

# the three best-checkpoint variants saved by train.py
CHECKPOINT_VARIANTS = ['best_auc', 'best_f1', 'best_loss']

def build_dataloaders(test_dataset, batch_size):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, collate_fn=emb_collate_fn)
    return test_dataloader

def import_model_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def build_model(model_cfg, device):
    ModelClass = import_model_class(model_cfg['module'], model_cfg['class_name'])

    model = ModelClass(model_cfg['params']).to(device)
    return model

def model_forward(model, embeddings, key_padding_mask):
    try:
        return model(embeddings, src_key_padding_mask=key_padding_mask)
    except TypeError:
        return model(embeddings)


def init_wandb(cfg, results_dir, train_run_id=None):
    """Initialize a W&B run for testing.

    If train_run_id is given (pulled from the loaded checkpoint), 
    resume that run so train and test metrics land in the SAME W&B run.
    Otherwise start a fresh run with job_type='test'.
    """
    wandb_cfg = cfg.get('wandb', {})
    if not wandb_cfg.get('enabled', False):
        return None

    training_cfg = cfg['training']
    default_name = f"{training_cfg['sub_dir']}_{training_cfg['checkpoint_prefix']}_test"

    run = wandb.init(
        project=wandb_cfg.get('project', 'oscar_nom_win'),
        entity=wandb_cfg.get('entity'),
        mode=wandb_cfg.get('mode', 'offline'),
        name=wandb_cfg.get('name') or default_name,
        notes=wandb_cfg.get('notes', training_cfg.get('notes', '')),
        tags=(wandb_cfg.get('tags', training_cfg.get('tags', [])) or []) + ["test"],
        config=cfg,
        dir=str(results_dir),
        id=train_run_id,
        resume="allow" if train_run_id else None,
        job_type="test",
    )
    return run

def evaluate_checkpoint(model, ckpt_path, test_dataloader, device,
                        use_amp, amp_dtype, batch_size, logger):
    """Load a checkpoint into `model`, run inference on the test set,
    return a dict of metrics + per-sample results.

    Uses the validation-tuned classification threshold stored in the checkpoint
    under 'val_threshold'. If absent (e.g. an older checkpoint), defaults to 0.5
    which is equivalent to the previous argmax behavior on softmax probs.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    threshold = float(checkpoint.get('val_threshold', 0.5))
    has_threshold = 'val_threshold' in checkpoint
    logger.info(
        f"Loaded weights from {ckpt_path.name}  "
        f"(threshold={threshold:.3f}{'' if has_threshold else ', DEFAULT — no val_threshold in ckpt'})"
    )

    model.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            imdb_ids   = batch['imdb_id']
            embeddings = batch['embeddings'].to(device)
            kpm        = batch['key_padding_mask'].to(device)
            targets    = batch['target'].to(device)

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model_forward(model, embeddings, kpm)

            logits = logits.float()

            probabilities = F.softmax(logits, dim=1)[:, 1]
            # threshold-based prediction (replaces argmax which was equivalent to thr=0.5)
            predictions = (probabilities >= threshold).long()

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            preds_list = predictions.cpu().numpy().tolist()
            targets_list = targets.cpu().numpy().tolist()
            probs_list = probabilities.cpu().numpy().tolist()

            all_predictions.extend(preds_list)
            all_targets.extend(targets_list)
            all_probabilities.extend(probs_list)

            for i in range(len(targets_list)):
                sample_idx = batch_idx * batch_size + i
                results.append({
                    "idx": sample_idx,
                    "imdb_id": imdb_ids[i],
                    "target": targets_list[i],
                    "model_prediction": preds_list[i],
                    "model_prob": round(probs_list[i], 6)
                })

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(test_dataloader)} batches")

    acc       = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='binary', pos_label=1, zero_division=0)
    recall    = recall_score(all_targets, all_predictions, average='binary', pos_label=1, zero_division=0)
    f1        = f1_score(all_targets, all_predictions, average='binary', pos_label=1, zero_division=0)
    macro_f1  = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    auc       = roc_auc_score(all_targets, all_probabilities)

    return {
        'metrics': {
            'accuracy':  acc,
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'macro_f1':  macro_f1,
            'auc':       auc,
            'correct':   correct,
            'total':     total,
            'threshold': threshold,
            'threshold_source': 'checkpoint' if has_threshold else 'default_0.5',
        },
        'all_targets':       all_targets,
        'all_predictions':   all_predictions,
        'all_probabilities': all_probabilities,
        'results':           results,
        'wandb_run_id':      checkpoint.get('wandb_run_id'),
    }

def main():
    # argument parsing
    parser = argparse.ArgumentParser(description='Oscar nomination prediction model test evaluator, embeddings input')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file')

    parser.add_argument('--new-wandb-run', action='store_true',
                    help="Force a fresh W&B run for this test instead of "
                    "resuming the training run stored in the checkpoint.")

    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='Batch size for test inference. Default 1 to avoid OOM '
                             'on long scripts. Override config training batch_size.')

    parser.add_argument('--no-amp', action='store_true',
                        help='Disable bfloat16 autocast during inference.')
    
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    data_cfg = cfg['data']
    model_cfg = cfg['model']
    training_cfg = cfg['training']

    checkpoint_prefix = training_cfg.get('checkpoint_prefix', 'model')

    models_dir = Path(training_cfg.get('models_dir', '../models')) / training_cfg['sub_dir']
    results_dir = Path(training_cfg.get('results_dir', '../results')) / training_cfg['sub_dir']
    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    # save terminal outputs to a log file in the results folder
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(results_dir / f"{training_cfg['checkpoint_prefix']}_test.log")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s • %(name)s: \n%(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # seeds for reproducibility
    seed = training_cfg.get('seed', 1337)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}\n')

    # decide AMP usage: bf16 on CUDA unless --no-amp passed
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    amp_dtype = torch.bfloat16
    logger.info(f"Inference AMP: {'bfloat16' if use_amp else 'disabled (fp32)'}")
    logger.info(f"Test batch size: {args.test_batch_size}")

    embed_dir = data_cfg['embed_dir']
    eager     = data_cfg.get('eager_load', True)
    test_dataset = OscarEmbeddingDataset(embed_dir, split='test', eager=eager)
    test_dataloader = build_dataloaders(test_dataset, args.test_batch_size)

    # build the model once; we'll load each checkpoint's weights in turn
    model = build_model(model_cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}\n")

    # -----------------------------------------------------------------
    # Evaluate each checkpoint variant in turn.
    # -----------------------------------------------------------------
    eval_outputs = {}        # variant -> evaluate_checkpoint() return dict
    missing_variants = []
    train_run_id = None      # grabbed from whichever checkpoint loads first

    for variant in CHECKPOINT_VARIANTS:
        ckpt_path = models_dir / f'{checkpoint_prefix}_{variant}.pth'
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
            missing_variants.append(variant)
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating variant: {variant}")
        logger.info(f"{'='*60}")

        out = evaluate_checkpoint(
            model, ckpt_path, test_dataloader, device,
            use_amp, amp_dtype, args.test_batch_size, logger,
        )
        eval_outputs[variant] = out

        if train_run_id is None and out['wandb_run_id'] is not None:
            train_run_id = out['wandb_run_id']

        m = out['metrics']
        logger.info(
            f"  [{variant}] thr={m['threshold']:.3f} ({m['threshold_source']})  "
            f"Acc={m['accuracy']*100:.2f}%  "
            f"P={m['precision']*100:.2f}%  R={m['recall']*100:.2f}%  "
            f"F1={m['f1']*100:.2f}%  macro-F1={m['macro_f1']*100:.2f}%  "
            f"AUC={m['auc']:.4f}"
        )

        # save per-variant predictions JSON
        results_path = results_dir / f"{checkpoint_prefix}_{variant}_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(out['results'], f, indent=4)
        logger.info(f"  Predictions saved to {results_path}")

    if not eval_outputs:
        logger.error("No checkpoints found to evaluate. Exiting.")
        return

    # -----------------------------------------------------------------
    # Pick the best variant by composite score = AUC only
    # Tie-break in CHECKPOINT_VARIANTS order.
    # -----------------------------------------------------------------
    composite_scores = {
        v: out['metrics']['auc']
        for v, out in eval_outputs.items()
    }

    logger.info(f"\n{'='*60}")
    logger.info("Composite score = AUC only")
    logger.info(f"{'='*60}")
    for v in CHECKPOINT_VARIANTS:
        if v in composite_scores:
            logger.info(f"  {v:10s}: {composite_scores[v]:.4f}")

    # argmax with deterministic tie-break: iterate in CHECKPOINT_VARIANTS order
    best_variant = max(
        (v for v in CHECKPOINT_VARIANTS if v in composite_scores),
        key=lambda v: composite_scores[v],
    )
    best_out = eval_outputs[best_variant]
    best_metrics = best_out['metrics']
    best_composite = composite_scores[best_variant]

    logger.info(f"\n{'='*60}")
    logger.info(f"BEST variant: {best_variant}  (composite={best_composite:.4f})")
    logger.info(f"{'='*60}")
    logger.info(f"  Threshold:  {best_metrics['threshold']:.3f}  ({best_metrics['threshold_source']})")
    logger.info(f"  Accuracy:   {best_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  Precision:  {best_metrics['precision'] * 100:.2f}%")
    logger.info(f"  Recall:     {best_metrics['recall'] * 100:.2f}%")
    logger.info(f"  F1:         {best_metrics['f1'] * 100:.2f}%")
    logger.info(f"  Macro-F1:   {best_metrics['macro_f1'] * 100:.2f}%")
    logger.info(f"  AUC:        {best_metrics['auc']:.4f}")
    logger.info(f"{'='*60}")

    # -----------------------------------------------------------------
    # W&B logging: report ONLY the best variant.
    # -----------------------------------------------------------------
    train_run_id = None if args.new_wandb_run else train_run_id
    wandb_run = init_wandb(cfg, results_dir, train_run_id=train_run_id)

    if wandb_run is not None:
        if train_run_id:
            logger.info(f"W&B: continuing run id={wandb_run.id} for test logging")
        else:
            logger.info(f"W&B: started fresh test run id={wandb_run.id}")

        try:
            # record which variant won + composite + all variant composites for context
            wandb.log({
                "test/best_variant":   best_variant,
                "test/composite":      best_composite,
                "test/threshold":      best_metrics['threshold'],
                "test/accuracy":       best_metrics['accuracy'],
                "test/precision":      best_metrics['precision'],
                "test/recall":         best_metrics['recall'],
                "test/f1":             best_metrics['f1'],
                "test/macro_f1":       best_metrics['macro_f1'],
                "test/auc":            best_metrics['auc'],
            })

            # also log every variant's composite so you can see the spread on W&B
            for v, score in composite_scores.items():
                wandb.log({f"test/composite_{v}": score})

            # W&B leaderboard summary metrics
            wandb.run.summary["test/best_variant"] = best_variant
            wandb.run.summary["test/composite"]    = best_composite
            wandb.run.summary["test/threshold"]    = best_metrics['threshold']
            wandb.run.summary["test/accuracy"]     = best_metrics['accuracy']
            wandb.run.summary["test/precision"]    = best_metrics['precision']
            wandb.run.summary["test/recall"]       = best_metrics['recall']
            wandb.run.summary["test/f1"]           = best_metrics['f1']
            wandb.run.summary["test/macro_f1"]     = best_metrics['macro_f1']
            wandb.run.summary["test/auc"]          = best_metrics['auc']
            if missing_variants:
                wandb.run.summary["test/missing_variants"] = ",".join(missing_variants)

            # confusion matrix for the best variant
            wandb.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=best_out['all_targets'],
                    preds=best_out['all_predictions'],
                    class_names=["Not Nominated", "Nominated"],
                )
            })

            # ROC curve for the best variant
            roc_probs = [[1.0 - p, p] for p in best_out['all_probabilities']]
            wandb.log({
                "test/roc": wandb.plot.roc_curve(
                    y_true=best_out['all_targets'],
                    y_probas=roc_probs,
                    labels=["Not Nominated", "Nominated"],
                )
            })

            # per-prediction table for the best variant
            pred_table = wandb.Table(
                columns=["idx", "imdb_id", "target", "prediction", "prob_nominated", "correct"],
                data=[[
                    r["idx"],
                    r["imdb_id"],
                    r["target"],
                    r["model_prediction"],
                    r["model_prob"],
                    int(r["target"] == r["model_prediction"]),
                ] for r in best_out['results']],
            )
            wandb.log({"test/predictions": pred_table})

            # small comparison table across all evaluated variants
            variant_table = wandb.Table(
                columns=["variant", "threshold", "composite", "accuracy", "auc", "f1",
                         "macro_f1", "precision", "recall", "is_best"],
                data=[[
                    v,
                    eval_outputs[v]['metrics']['threshold'],
                    composite_scores[v],
                    eval_outputs[v]['metrics']['accuracy'],
                    eval_outputs[v]['metrics']['auc'],
                    eval_outputs[v]['metrics']['f1'],
                    eval_outputs[v]['metrics']['macro_f1'],
                    eval_outputs[v]['metrics']['precision'],
                    eval_outputs[v]['metrics']['recall'],
                    int(v == best_variant),
                ] for v in CHECKPOINT_VARIANTS if v in eval_outputs],
            )
            wandb.log({"test/variant_comparison": variant_table})

        finally:
            wandb.finish()

if __name__ == "__main__":
    main()