import argparse
import json
import pickle
import importlib
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from datasets import OscarScriptDataset

def build_dataloaders(test_dataset, training_cfg):
    bs = training_cfg['batch_size']
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=bs,
                                 shuffle=False, num_workers=2)
    return test_dataloader

def import_model_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def build_model(model_cfg, device):
    ModelClass = import_model_class(model_cfg['module'], model_cfg['class_name'])

    model = ModelClass(model_cfg['params']).to(device)
    return model

def main():
    # argument parsing
    parser = argparse.ArgumentParser(description='Oscar nomination prediction model test evaluator')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file')
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

    with open(data_cfg['test_path'], 'rb') as f:
        test_items = pickle.load(f)

    max_seq_len = model_cfg['params']['max_seq_len']
    
    test_dataset = OscarScriptDataset(test_items, max_length=max_seq_len)
    test_dataloader = build_dataloaders(test_dataset, training_cfg)

    # define model
    
    model = build_model(model_cfg, device)
    best_path = models_dir / f'{checkpoint_prefix}_best.pth'
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model weights loaded successfully { best_path}!")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    logger.info("Evaluating model on test set...")
    model.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []  # store probabilities for ROC curve
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            imdb_ids = batch['imdb_id']
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            # Forward pass - get model predictions
            logits = model(input_ids)

            # Get predicted class (0 or 1)
            predictions = torch.argmax(logits, dim=1)

            # Get probabilities for positive class (nominated = 1)
            probabilities = F.softmax(logits, dim=1)[:, 1]  # Probability of class 1

            # Update accuracy metrics
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            # Store predictions, targets, and probabilities
            preds_list = predictions.cpu().numpy().tolist()
            targets_list = targets.cpu().numpy().tolist()
            probs_list = probabilities.cpu().numpy().tolist()

            all_predictions.extend(preds_list)
            all_targets.extend(targets_list)
            all_probabilities.extend(probs_list)

            for i in range(len(targets_list)):
                sample_idx = batch_idx * training_cfg['batch_size'] + i
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
    precision = precision_score(all_targets, all_predictions, average='binary', pos_label=1)
    recall    = recall_score(all_targets, all_predictions, average='binary', pos_label=1)
    f1        = f1_score(all_targets, all_predictions, average='binary', pos_label=1)
    macro_f1  = f1_score(all_targets, all_predictions, average='macro')
    auc       = roc_auc_score(all_targets, all_probabilities)

    logger.info(f"\n{'='*60}")
    logger.info(f"Test Results ({correct}/{total} correct)")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy:   {acc * 100:.2f}%")
    logger.info(f"  Precision:  {precision * 100:.2f}%")
    logger.info(f"  Recall:     {recall * 100:.2f}%")
    logger.info(f"  F1:         {f1 * 100:.2f}%")
    logger.info(f"  Macro-F1:   {macro_f1 * 100:.2f}%")
    logger.info(f"  AUC:        {auc:.4f}")
    logger.info(f"{'='*60}")

    # save results JSON to same directory as model checkpoint
    results_path = results_dir / f"{checkpoint_prefix}_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nTest dataset predictions saved to {results_path}")

if __name__ == "__main__":
    main()