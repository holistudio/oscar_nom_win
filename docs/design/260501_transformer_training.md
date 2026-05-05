# Mean-Pool Transformer Model Training

All models were trained under the same training parameters

```json
"training": {
    "epochs": 15,
    "batch_size": 4,
    "peak_lr": 0.0001,
    "weight_decay": 0.1,
    "eta_min": 1e-06,
    "warmup_fraction": 0.1,
    "grad_clip": 1.0,
    "class_weights": [
      1.0,
      8.0
    ],
    "label_smoothing": 0.0,
    "seed": 1337,
  }
```

## Key takeaways:

1. The model is overfitting violently and you're saving the overfit checkpoint.
2. Best-checkpoint instability is real. You're discarding signal by only tracking val AUC for selection.
3. Save multiple best validation checkpoints: `best_auc, best_f1, best_loss`. Then test all and see which performs best.
4. Ideally, report test metrics for all three versions, but `best_test_AUC` version is probably the one to focus on.
5. Class weights `[1, 8]` may be too aggressive.
6. 15 epochs is too many, consider using early stopping.

## Training Logs

The following training logs were recorded for various params:

### start

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 128,
      "enc_nhead": 4,
      "enc_num_layers": 2,
      "agg_d_model": 128,
      "agg_nhead": 4,
      "agg_num_layers": 2,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
14:22:53 • root: 
Epoch [1/15] - Train Loss: 0.7476, Val Loss: 0.7405, Train Acc: 25.2%, Val Acc: 19.1%, Val AUC: 0.648, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:54, Avg/Epoch: 00:00:54 - New best! Model saved.
14:23:47 • root: 
Epoch [2/15] - Train Loss: 0.7765, Val Loss: 0.8282, Train Acc: 21.4%, Val Acc: 19.1%, Val AUC: 0.640, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:49, Avg/Epoch: 00:00:54
14:24:42 • root: 
Epoch [3/15] - Train Loss: 0.7456, Val Loss: 0.6973, Train Acc: 34.0%, Val Acc: 80.7%, Val AUC: 0.492, Val P/R/F1: 0.33/0.01/0.02 - Elapsed: 00:02:43, Avg/Epoch: 00:00:54
14:25:37 • root: 
Epoch [4/15] - Train Loss: 0.7310, Val Loss: 0.9161, Train Acc: 68.5%, Val Acc: 80.9%, Val AUC: 0.718, Val P/R/F1: 0.50/0.08/0.14 - Elapsed: 00:03:38, Avg/Epoch: 00:00:54 - New best! Model saved.
14:26:31 • root: 
Epoch [5/15] - Train Loss: 0.7667, Val Loss: 1.8844, Train Acc: 80.2%, Val Acc: 80.2%, Val AUC: 0.720, Val P/R/F1: 0.20/0.01/0.02 - Elapsed: 00:04:32, Avg/Epoch: 00:00:54 - New best! Model saved.
14:27:27 • root: 
Epoch [6/15] - Train Loss: 0.6336, Val Loss: 1.7896, Train Acc: 88.3%, Val Acc: 80.0%, Val AUC: 0.743, Val P/R/F1: 0.43/0.14/0.21 - Elapsed: 00:05:27, Avg/Epoch: 00:00:54 - New best! Model saved.
14:28:22 • root: 
Epoch [7/15] - Train Loss: 0.4358, Val Loss: 2.5087, Train Acc: 94.2%, Val Acc: 79.3%, Val AUC: 0.712, Val P/R/F1: 0.36/0.11/0.17 - Elapsed: 00:06:23, Avg/Epoch: 00:00:54
14:29:15 • root: 
Epoch [8/15] - Train Loss: 0.2811, Val Loss: 2.9370, Train Acc: 96.5%, Val Acc: 80.0%, Val AUC: 0.713, Val P/R/F1: 0.39/0.08/0.14 - Elapsed: 00:07:17, Avg/Epoch: 00:00:54
14:30:11 • root: 
Epoch [9/15] - Train Loss: 0.2327, Val Loss: 2.9690, Train Acc: 97.5%, Val Acc: 79.5%, Val AUC: 0.662, Val P/R/F1: 0.38/0.12/0.18 - Elapsed: 00:08:12, Avg/Epoch: 00:00:54
14:31:06 • root: 
Epoch [10/15] - Train Loss: 0.0744, Val Loss: 3.0964, Train Acc: 99.0%, Val Acc: 79.8%, Val AUC: 0.651, Val P/R/F1: 0.42/0.17/0.24 - Elapsed: 00:09:07, Avg/Epoch: 00:00:54
14:32:00 • root: 
Epoch [11/15] - Train Loss: 0.0235, Val Loss: 3.2681, Train Acc: 99.8%, Val Acc: 80.0%, Val AUC: 0.631, Val P/R/F1: 0.40/0.10/0.15 - Elapsed: 00:10:01, Avg/Epoch: 00:00:54
14:32:55 • root: 
Epoch [12/15] - Train Loss: 0.0347, Val Loss: 3.2013, Train Acc: 99.7%, Val Acc: 79.3%, Val AUC: 0.606, Val P/R/F1: 0.42/0.21/0.28 - Elapsed: 00:10:56, Avg/Epoch: 00:00:54
14:33:49 • root: 
Epoch [13/15] - Train Loss: 0.0061, Val Loss: 3.0234, Train Acc: 99.9%, Val Acc: 80.0%, Val AUC: 0.617, Val P/R/F1: 0.43/0.15/0.23 - Elapsed: 00:11:50, Avg/Epoch: 00:00:54
14:34:44 • root: 
Epoch [14/15] - Train Loss: 0.0159, Val Loss: 3.3712, Train Acc: 99.9%, Val Acc: 79.8%, Val AUC: 0.634, Val P/R/F1: 0.40/0.12/0.18 - Elapsed: 00:12:45, Avg/Epoch: 00:00:54
14:35:38 • root: 
Epoch [15/15] - Train Loss: 0.0101, Val Loss: 3.4726, Train Acc: 99.9%, Val Acc: 80.0%, Val AUC: 0.613, Val P/R/F1: 0.42/0.12/0.19 - Elapsed: 00:13:40, Avg/Epoch: 00:00:54
14:35:38 • root: 

Training complete!
```

### heads

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 128,
      "enc_nhead": 8,
      "enc_num_layers": 2,
      "agg_d_model": 128,
      "agg_nhead": 8,
      "agg_num_layers": 2,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
14:37:13 • root: 
Epoch [1/15] - Train Loss: 0.7486, Val Loss: 0.7446, Train Acc: 24.6%, Val Acc: 19.1%, Val AUC: 0.627, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:05, Avg/Epoch: 00:01:05 - New best! Model saved.
14:38:19 • root: 
Epoch [2/15] - Train Loss: 0.7762, Val Loss: 0.8336, Train Acc: 21.4%, Val Acc: 19.1%, Val AUC: 0.652, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:02:11, Avg/Epoch: 00:01:05 - New best! Model saved.
14:39:24 • root: 
Epoch [3/15] - Train Loss: 0.7441, Val Loss: 0.6894, Train Acc: 33.0%, Val Acc: 80.2%, Val AUC: 0.508, Val P/R/F1: 0.40/0.07/0.12 - Elapsed: 00:03:17, Avg/Epoch: 00:01:05
14:40:31 • root: 
Epoch [4/15] - Train Loss: 0.7088, Val Loss: 1.2364, Train Acc: 70.8%, Val Acc: 81.4%, Val AUC: 0.733, Val P/R/F1: 0.62/0.06/0.11 - Elapsed: 00:04:23, Avg/Epoch: 00:01:05 - New best! Model saved.
14:41:37 • root: 
Epoch [5/15] - Train Loss: 0.8358, Val Loss: 2.0802, Train Acc: 82.0%, Val Acc: 79.3%, Val AUC: 0.739, Val P/R/F1: 0.23/0.04/0.06 - Elapsed: 00:05:29, Avg/Epoch: 00:01:05 - New best! Model saved.
14:42:42 • root: 
Epoch [6/15] - Train Loss: 0.5697, Val Loss: 2.3826, Train Acc: 91.1%, Val Acc: 79.3%, Val AUC: 0.722, Val P/R/F1: 0.37/0.12/0.18 - Elapsed: 00:06:36, Avg/Epoch: 00:01:05
14:43:48 • root: 
Epoch [7/15] - Train Loss: 0.3641, Val Loss: 2.8459, Train Acc: 95.5%, Val Acc: 78.6%, Val AUC: 0.698, Val P/R/F1: 0.22/0.05/0.08 - Elapsed: 00:07:41, Avg/Epoch: 00:01:05
14:44:54 • root: 
Epoch [8/15] - Train Loss: 0.2581, Val Loss: 2.5688, Train Acc: 97.1%, Val Acc: 75.9%, Val AUC: 0.709, Val P/R/F1: 0.36/0.32/0.34 - Elapsed: 00:08:46, Avg/Epoch: 00:01:05
14:45:59 • root: 
Epoch [9/15] - Train Loss: 0.1467, Val Loss: 3.1003, Train Acc: 98.4%, Val Acc: 78.9%, Val AUC: 0.679, Val P/R/F1: 0.36/0.14/0.21 - Elapsed: 00:09:52, Avg/Epoch: 00:01:05
14:47:04 • root: 
Epoch [10/15] - Train Loss: 0.0590, Val Loss: 3.0282, Train Acc: 99.3%, Val Acc: 77.7%, Val AUC: 0.658, Val P/R/F1: 0.36/0.21/0.27 - Elapsed: 00:10:58, Avg/Epoch: 00:01:05
14:48:10 • root: 
Epoch [11/15] - Train Loss: 0.0447, Val Loss: 3.1725, Train Acc: 99.5%, Val Acc: 80.0%, Val AUC: 0.688, Val P/R/F1: 0.42/0.13/0.20 - Elapsed: 00:12:03, Avg/Epoch: 00:01:05
14:49:16 • root: 
Epoch [12/15] - Train Loss: 0.0359, Val Loss: 3.4672, Train Acc: 99.5%, Val Acc: 79.1%, Val AUC: 0.651, Val P/R/F1: 0.38/0.14/0.21 - Elapsed: 00:13:09, Avg/Epoch: 00:01:05
14:50:22 • root: 
Epoch [13/15] - Train Loss: 0.0335, Val Loss: 3.0700, Train Acc: 99.5%, Val Acc: 79.1%, Val AUC: 0.633, Val P/R/F1: 0.38/0.14/0.21 - Elapsed: 00:14:15, Avg/Epoch: 00:01:05
14:51:27 • root: 
Epoch [14/15] - Train Loss: 0.0420, Val Loss: 3.3464, Train Acc: 99.7%, Val Acc: 79.5%, Val AUC: 0.717, Val P/R/F1: 0.39/0.13/0.20 - Elapsed: 00:15:20, Avg/Epoch: 00:01:05
14:52:33 • root: 
Epoch [15/15] - Train Loss: 0.0364, Val Loss: 3.4283, Train Acc: 99.7%, Val Acc: 80.2%, Val AUC: 0.657, Val P/R/F1: 0.44/0.13/0.20 - Elapsed: 00:16:26, Avg/Epoch: 00:01:05
14:52:33 • root: 

Training complete!
```

### wide

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 160,
      "enc_nhead": 4,
      "enc_num_layers": 2,
      "agg_d_model": 160,
      "agg_nhead": 4,
      "agg_num_layers": 2,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
14:54:17 • root: 
Epoch [1/15] - Train Loss: 0.7814, Val Loss: 0.8391, Train Acc: 27.3%, Val Acc: 19.1%, Val AUC: 0.625, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:14, Avg/Epoch: 00:01:14 - New best! Model saved.
14:55:33 • root: 
Epoch [2/15] - Train Loss: 0.7734, Val Loss: 0.7677, Train Acc: 37.8%, Val Acc: 19.1%, Val AUC: 0.597, Val P/R/F1: 0.19/0.99/0.32 - Elapsed: 00:02:30, Avg/Epoch: 00:01:15
14:56:48 • root: 
Epoch [3/15] - Train Loss: 0.7473, Val Loss: 0.6601, Train Acc: 22.8%, Val Acc: 58.4%, Val AUC: 0.675, Val P/R/F1: 0.28/0.75/0.41 - Elapsed: 00:03:45, Avg/Epoch: 00:01:14 - New best! Model saved.
14:58:02 • root: 
Epoch [4/15] - Train Loss: 0.7807, Val Loss: 1.1067, Train Acc: 64.7%, Val Acc: 80.9%, Val AUC: 0.590, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:05:00, Avg/Epoch: 00:01:14
14:59:18 • root: 
Epoch [5/15] - Train Loss: 0.7790, Val Loss: 1.3241, Train Acc: 77.5%, Val Acc: 81.4%, Val AUC: 0.724, Val P/R/F1: 0.56/0.12/0.20 - Elapsed: 00:06:16, Avg/Epoch: 00:01:14 - New best! Model saved.
15:00:32 • root: 
Epoch [6/15] - Train Loss: 0.7015, Val Loss: 1.8788, Train Acc: 86.3%, Val Acc: 80.5%, Val AUC: 0.733, Val P/R/F1: 0.47/0.23/0.31 - Elapsed: 00:07:30, Avg/Epoch: 00:01:14 - New best! Model saved.
15:01:48 • root: 
Epoch [7/15] - Train Loss: 0.5641, Val Loss: 2.0640, Train Acc: 89.6%, Val Acc: 78.4%, Val AUC: 0.733, Val P/R/F1: 0.40/0.27/0.33 - Elapsed: 00:08:45, Avg/Epoch: 00:01:14 - New best! Model saved.
15:03:02 • root: 
Epoch [8/15] - Train Loss: 0.4001, Val Loss: 2.1959, Train Acc: 93.6%, Val Acc: 80.9%, Val AUC: 0.733, Val P/R/F1: 0.50/0.24/0.32 - Elapsed: 00:10:00, Avg/Epoch: 00:01:14
15:04:18 • root: 
Epoch [9/15] - Train Loss: 0.2776, Val Loss: 2.5971, Train Acc: 96.7%, Val Acc: 80.5%, Val AUC: 0.726, Val P/R/F1: 0.48/0.25/0.33 - Elapsed: 00:11:16, Avg/Epoch: 00:01:14
15:05:33 • root: 
Epoch [10/15] - Train Loss: 0.1641, Val Loss: 2.3949, Train Acc: 98.0%, Val Acc: 79.1%, Val AUC: 0.724, Val P/R/F1: 0.44/0.32/0.37 - Elapsed: 00:12:30, Avg/Epoch: 00:01:14
15:06:47 • root: 
Epoch [11/15] - Train Loss: 0.1402, Val Loss: 2.3029, Train Acc: 98.6%, Val Acc: 72.3%, Val AUC: 0.717, Val P/R/F1: 0.35/0.52/0.42 - Elapsed: 00:13:45, Avg/Epoch: 00:01:14
15:08:01 • root: 
Epoch [12/15] - Train Loss: 0.0568, Val Loss: 2.5962, Train Acc: 99.2%, Val Acc: 78.6%, Val AUC: 0.717, Val P/R/F1: 0.42/0.33/0.37 - Elapsed: 00:14:59, Avg/Epoch: 00:01:14
15:09:16 • root: 
Epoch [13/15] - Train Loss: 0.0594, Val Loss: 2.6560, Train Acc: 99.1%, Val Acc: 78.6%, Val AUC: 0.730, Val P/R/F1: 0.42/0.32/0.36 - Elapsed: 00:16:13, Avg/Epoch: 00:01:14
15:10:32 • root: 
Epoch [14/15] - Train Loss: 0.0372, Val Loss: 2.5487, Train Acc: 99.5%, Val Acc: 75.9%, Val AUC: 0.717, Val P/R/F1: 0.39/0.44/0.41 - Elapsed: 00:17:30, Avg/Epoch: 00:01:14
15:11:46 • root: 
Epoch [15/15] - Train Loss: 0.0340, Val Loss: 2.6079, Train Acc: 99.5%, Val Acc: 75.9%, Val AUC: 0.724, Val P/R/F1: 0.38/0.42/0.40 - Elapsed: 00:18:44, Avg/Epoch: 00:01:14
15:11:46 • root: 

Training complete!
```

### deep

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 128,
      "enc_nhead": 4,
      "enc_num_layers": 4,
      "agg_d_model": 128,
      "agg_nhead": 4,
      "agg_num_layers": 4,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
15:13:59 • root: 
Epoch [1/15] - Train Loss: 0.7603, Val Loss: 0.7226, Train Acc: 31.3%, Val Acc: 19.1%, Val AUC: 0.620, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:42, Avg/Epoch: 00:01:42 - New best! Model saved.
15:15:41 • root: 
Epoch [2/15] - Train Loss: 0.7695, Val Loss: 0.8730, Train Acc: 30.7%, Val Acc: 19.1%, Val AUC: 0.587, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:03:25, Avg/Epoch: 00:01:42
15:17:24 • root: 
Epoch [3/15] - Train Loss: 0.7595, Val Loss: 0.7750, Train Acc: 29.4%, Val Acc: 19.1%, Val AUC: 0.626, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:05:07, Avg/Epoch: 00:01:42 - New best! Model saved.
15:19:07 • root: 
Epoch [4/15] - Train Loss: 0.7043, Val Loss: 0.7471, Train Acc: 50.3%, Val Acc: 80.9%, Val AUC: 0.687, Val P/R/F1: 0.50/0.26/0.34 - Elapsed: 00:06:50, Avg/Epoch: 00:01:41 - New best! Model saved.
15:20:50 • root: 
Epoch [5/15] - Train Loss: 0.7572, Val Loss: 1.7633, Train Acc: 82.6%, Val Acc: 80.9%, Val AUC: 0.714, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:08:32, Avg/Epoch: 00:01:41 - New best! Model saved.
15:22:32 • root: 
Epoch [6/15] - Train Loss: 0.7381, Val Loss: 1.5250, Train Acc: 86.4%, Val Acc: 79.3%, Val AUC: 0.711, Val P/R/F1: 0.43/0.27/0.34 - Elapsed: 00:10:16, Avg/Epoch: 00:01:41
15:24:15 • root: 
Epoch [7/15] - Train Loss: 0.5367, Val Loss: 2.0834, Train Acc: 91.0%, Val Acc: 80.9%, Val AUC: 0.662, Val P/R/F1: 0.50/0.15/0.24 - Elapsed: 00:11:58, Avg/Epoch: 00:01:41
15:25:58 • root: 
Epoch [8/15] - Train Loss: 0.3867, Val Loss: 2.2156, Train Acc: 94.5%, Val Acc: 80.5%, Val AUC: 0.604, Val P/R/F1: 0.47/0.17/0.25 - Elapsed: 00:13:40, Avg/Epoch: 00:01:41
15:27:40 • root: 
Epoch [9/15] - Train Loss: 0.1965, Val Loss: 1.9577, Train Acc: 97.0%, Val Acc: 76.8%, Val AUC: 0.675, Val P/R/F1: 0.40/0.42/0.41 - Elapsed: 00:15:23, Avg/Epoch: 00:01:41
15:29:22 • root: 
Epoch [10/15] - Train Loss: 0.1167, Val Loss: 2.3709, Train Acc: 98.5%, Val Acc: 79.3%, Val AUC: 0.662, Val P/R/F1: 0.44/0.30/0.35 - Elapsed: 00:17:06, Avg/Epoch: 00:01:41
15:31:04 • root: 
Epoch [11/15] - Train Loss: 0.0653, Val Loss: 2.3264, Train Acc: 99.2%, Val Acc: 80.9%, Val AUC: 0.642, Val P/R/F1: 0.50/0.29/0.36 - Elapsed: 00:18:48, Avg/Epoch: 00:01:41
15:32:46 • root: 
Epoch [12/15] - Train Loss: 0.0480, Val Loss: 2.3403, Train Acc: 99.5%, Val Acc: 78.2%, Val AUC: 0.662, Val P/R/F1: 0.40/0.30/0.34 - Elapsed: 00:20:30, Avg/Epoch: 00:01:41
15:34:28 • root: 
Epoch [13/15] - Train Loss: 0.0369, Val Loss: 2.4125, Train Acc: 99.7%, Val Acc: 78.6%, Val AUC: 0.636, Val P/R/F1: 0.42/0.31/0.36 - Elapsed: 00:22:12, Avg/Epoch: 00:01:41
15:36:10 • root: 
Epoch [14/15] - Train Loss: 0.0311, Val Loss: 2.5485, Train Acc: 99.7%, Val Acc: 79.1%, Val AUC: 0.656, Val P/R/F1: 0.43/0.29/0.34 - Elapsed: 00:23:54, Avg/Epoch: 00:01:41
15:37:52 • root: 
Epoch [15/15] - Train Loss: 0.0309, Val Loss: 2.5276, Train Acc: 99.8%, Val Acc: 79.1%, Val AUC: 0.627, Val P/R/F1: 0.43/0.31/0.36 - Elapsed: 00:25:36, Avg/Epoch: 00:01:41
15:37:52 • root: 

Training complete!
```

### wide_deep

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 160,
      "enc_nhead": 4,
      "enc_num_layers": 4,
      "agg_d_model": 160,
      "agg_nhead": 4,
      "agg_num_layers": 4,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
15:40:45 • root: 
Epoch [1/15] - Train Loss: 0.7573, Val Loss: 0.7434, Train Acc: 43.3%, Val Acc: 19.1%, Val AUC: 0.591, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:02:19, Avg/Epoch: 00:02:19 - New best! Model saved.
15:43:04 • root: 
Epoch [2/15] - Train Loss: 0.7686, Val Loss: 0.6753, Train Acc: 35.5%, Val Acc: 80.2%, Val AUC: 0.584, Val P/R/F1: 0.40/0.07/0.12 - Elapsed: 00:04:39, Avg/Epoch: 00:02:19
15:45:23 • root: 
Epoch [3/15] - Train Loss: 0.7614, Val Loss: 0.7325, Train Acc: 30.3%, Val Acc: 19.3%, Val AUC: 0.590, Val P/R/F1: 0.19/0.98/0.32 - Elapsed: 00:06:58, Avg/Epoch: 00:02:18
15:47:43 • root: 
Epoch [4/15] - Train Loss: 0.7247, Val Loss: 0.9390, Train Acc: 41.8%, Val Acc: 19.1%, Val AUC: 0.610, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:09:17, Avg/Epoch: 00:02:18 - New best! Model saved.
15:50:02 • root: 
Epoch [5/15] - Train Loss: 0.7565, Val Loss: 0.6951, Train Acc: 23.6%, Val Acc: 80.7%, Val AUC: 0.601, Val P/R/F1: 0.46/0.07/0.12 - Elapsed: 00:11:37, Avg/Epoch: 00:02:18
15:52:21 • root: 
Epoch [6/15] - Train Loss: 0.8599, Val Loss: 1.0208, Train Acc: 77.6%, Val Acc: 78.0%, Val AUC: 0.712, Val P/R/F1: 0.40/0.32/0.36 - Elapsed: 00:13:55, Avg/Epoch: 00:02:18 - New best! Model saved.
15:54:41 • root: 
Epoch [7/15] - Train Loss: 0.7132, Val Loss: 1.4260, Train Acc: 85.5%, Val Acc: 76.6%, Val AUC: 0.720, Val P/R/F1: 0.37/0.32/0.34 - Elapsed: 00:16:15, Avg/Epoch: 00:02:18 - New best! Model saved.
15:57:00 • root: 
Epoch [8/15] - Train Loss: 0.5140, Val Loss: 1.8116, Train Acc: 92.6%, Val Acc: 78.9%, Val AUC: 0.698, Val P/R/F1: 0.41/0.24/0.30 - Elapsed: 00:18:35, Avg/Epoch: 00:02:18
15:59:19 • root: 
Epoch [9/15] - Train Loss: 0.3350, Val Loss: 2.1706, Train Acc: 95.2%, Val Acc: 79.3%, Val AUC: 0.700, Val P/R/F1: 0.40/0.17/0.24 - Elapsed: 00:20:54, Avg/Epoch: 00:02:18
16:01:38 • root: 
Epoch [10/15] - Train Loss: 0.2830, Val Loss: 2.1227, Train Acc: 96.4%, Val Acc: 78.6%, Val AUC: 0.676, Val P/R/F1: 0.40/0.23/0.29 - Elapsed: 00:23:13, Avg/Epoch: 00:02:18
16:03:57 • root: 
Epoch [11/15] - Train Loss: 0.2135, Val Loss: 2.0747, Train Acc: 97.7%, Val Acc: 78.4%, Val AUC: 0.687, Val P/R/F1: 0.40/0.25/0.31 - Elapsed: 00:25:32, Avg/Epoch: 00:02:18
16:06:17 • root: 
Epoch [12/15] - Train Loss: 0.1410, Val Loss: 2.1286, Train Acc: 98.3%, Val Acc: 75.7%, Val AUC: 0.687, Val P/R/F1: 0.37/0.38/0.37 - Elapsed: 00:27:52, Avg/Epoch: 00:02:18
16:08:36 • root: 
Epoch [13/15] - Train Loss: 0.1272, Val Loss: 2.2803, Train Acc: 98.8%, Val Acc: 75.9%, Val AUC: 0.676, Val P/R/F1: 0.35/0.31/0.33 - Elapsed: 00:30:11, Avg/Epoch: 00:02:18
16:10:56 • root: 
Epoch [14/15] - Train Loss: 0.1047, Val Loss: 2.1916, Train Acc: 99.0%, Val Acc: 76.6%, Val AUC: 0.666, Val P/R/F1: 0.36/0.30/0.33 - Elapsed: 00:32:30, Avg/Epoch: 00:02:18
16:13:15 • root: 
Epoch [15/15] - Train Loss: 0.0813, Val Loss: 2.4010, Train Acc: 98.9%, Val Acc: 77.7%, Val AUC: 0.676, Val P/R/F1: 0.38/0.26/0.31 - Elapsed: 00:34:50, Avg/Epoch: 00:02:18
16:13:15 • root: 

Training complete!
```

### wide_heads

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 160,
      "enc_nhead": 8,
      "enc_num_layers": 2,
      "agg_d_model": 160,
      "agg_nhead": 8,
      "agg_num_layers": 2,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
16:15:18 • root: 
Epoch [1/15] - Train Loss: 0.7828, Val Loss: 0.8449, Train Acc: 27.0%, Val Acc: 19.1%, Val AUC: 0.648, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:25, Avg/Epoch: 00:01:25 - New best! Model saved.
16:16:43 • root: 
Epoch [2/15] - Train Loss: 0.7754, Val Loss: 0.7681, Train Acc: 35.5%, Val Acc: 19.3%, Val AUC: 0.599, Val P/R/F1: 0.19/0.99/0.32 - Elapsed: 00:02:51, Avg/Epoch: 00:01:25
16:18:08 • root: 
Epoch [3/15] - Train Loss: 0.7477, Val Loss: 0.6991, Train Acc: 23.2%, Val Acc: 38.4%, Val AUC: 0.670, Val P/R/F1: 0.22/0.88/0.35 - Elapsed: 00:04:16, Avg/Epoch: 00:01:24 - New best! Model saved.
16:19:34 • root: 
Epoch [4/15] - Train Loss: 0.8180, Val Loss: 1.2470, Train Acc: 65.2%, Val Acc: 80.9%, Val AUC: 0.584, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:05:42, Avg/Epoch: 00:01:24
16:20:59 • root: 
Epoch [5/15] - Train Loss: 0.8355, Val Loss: 1.0615, Train Acc: 76.3%, Val Acc: 81.1%, Val AUC: 0.723, Val P/R/F1: 0.52/0.19/0.28 - Elapsed: 00:07:07, Avg/Epoch: 00:01:24 - New best! Model saved.
16:22:25 • root: 
Epoch [6/15] - Train Loss: 0.7451, Val Loss: 1.3641, Train Acc: 85.0%, Val Acc: 73.4%, Val AUC: 0.737, Val P/R/F1: 0.37/0.55/0.44 - Elapsed: 00:08:32, Avg/Epoch: 00:01:24 - New best! Model saved.
16:23:49 • root: 
Epoch [7/15] - Train Loss: 0.5812, Val Loss: 2.1316, Train Acc: 90.3%, Val Acc: 79.5%, Val AUC: 0.731, Val P/R/F1: 0.41/0.15/0.22 - Elapsed: 00:09:57, Avg/Epoch: 00:01:24
16:25:15 • root: 
Epoch [8/15] - Train Loss: 0.4291, Val Loss: 2.0171, Train Acc: 93.6%, Val Acc: 74.5%, Val AUC: 0.725, Val P/R/F1: 0.36/0.43/0.39 - Elapsed: 00:11:23, Avg/Epoch: 00:01:24
16:26:40 • root: 
Epoch [9/15] - Train Loss: 0.2922, Val Loss: 2.4285, Train Acc: 95.5%, Val Acc: 76.4%, Val AUC: 0.716, Val P/R/F1: 0.37/0.33/0.35 - Elapsed: 00:12:48, Avg/Epoch: 00:01:24
16:28:06 • root: 
Epoch [10/15] - Train Loss: 0.1712, Val Loss: 2.3979, Train Acc: 97.8%, Val Acc: 77.3%, Val AUC: 0.702, Val P/R/F1: 0.39/0.33/0.36 - Elapsed: 00:14:13, Avg/Epoch: 00:01:24
16:29:30 • root: 
Epoch [11/15] - Train Loss: 0.1094, Val Loss: 2.3363, Train Acc: 98.8%, Val Acc: 71.6%, Val AUC: 0.708, Val P/R/F1: 0.34/0.54/0.42 - Elapsed: 00:15:38, Avg/Epoch: 00:01:24
16:30:56 • root: 
Epoch [12/15] - Train Loss: 0.0939, Val Loss: 2.5298, Train Acc: 98.9%, Val Acc: 77.0%, Val AUC: 0.715, Val P/R/F1: 0.40/0.38/0.39 - Elapsed: 00:17:04, Avg/Epoch: 00:01:24
16:32:21 • root: 
Epoch [13/15] - Train Loss: 0.0685, Val Loss: 2.5847, Train Acc: 99.3%, Val Acc: 76.6%, Val AUC: 0.723, Val P/R/F1: 0.37/0.33/0.35 - Elapsed: 00:18:29, Avg/Epoch: 00:01:24
16:33:46 • root: 
Epoch [14/15] - Train Loss: 0.0517, Val Loss: 2.5994, Train Acc: 99.4%, Val Acc: 74.1%, Val AUC: 0.713, Val P/R/F1: 0.35/0.40/0.37 - Elapsed: 00:19:54, Avg/Epoch: 00:01:24
16:35:13 • root: 
Epoch [15/15] - Train Loss: 0.0643, Val Loss: 2.6904, Train Acc: 99.4%, Val Acc: 74.1%, Val AUC: 0.718, Val P/R/F1: 0.35/0.40/0.37 - Elapsed: 00:21:20, Avg/Epoch: 00:01:24
16:35:13 • root: 

Training complete!
```

### heads_deep

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 128,
      "enc_nhead": 8,
      "enc_num_layers": 4,
      "agg_d_model": 128,
      "agg_nhead": 8,
      "agg_num_layers": 4,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
16:37:52 • root: 
Epoch [1/15] - Train Loss: 0.7610, Val Loss: 0.7186, Train Acc: 32.0%, Val Acc: 19.1%, Val AUC: 0.621, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:02:03, Avg/Epoch: 00:02:03 - New best! Model saved.
16:39:57 • root: 
Epoch [2/15] - Train Loss: 0.7680, Val Loss: 0.8651, Train Acc: 31.1%, Val Acc: 19.1%, Val AUC: 0.588, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:04:08, Avg/Epoch: 00:02:04
16:41:59 • root: 
Epoch [3/15] - Train Loss: 0.7616, Val Loss: 0.7590, Train Acc: 27.6%, Val Acc: 19.3%, Val AUC: 0.608, Val P/R/F1: 0.19/0.99/0.32 - Elapsed: 00:06:11, Avg/Epoch: 00:02:03
16:44:06 • root: 
Epoch [4/15] - Train Loss: 0.7176, Val Loss: 0.7574, Train Acc: 45.0%, Val Acc: 80.9%, Val AUC: 0.668, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:08:15, Avg/Epoch: 00:02:03 - New best! Model saved.
16:46:11 • root: 
Epoch [5/15] - Train Loss: 0.7830, Val Loss: 1.5028, Train Acc: 78.4%, Val Acc: 80.7%, Val AUC: 0.708, Val P/R/F1: 0.43/0.04/0.07 - Elapsed: 00:10:22, Avg/Epoch: 00:02:03 - New best! Model saved.
16:48:16 • root: 
Epoch [6/15] - Train Loss: 0.7485, Val Loss: 1.3377, Train Acc: 85.8%, Val Acc: 78.9%, Val AUC: 0.737, Val P/R/F1: 0.44/0.38/0.41 - Elapsed: 00:12:27, Avg/Epoch: 00:02:03 - New best! Model saved.
16:50:18 • root: 
Epoch [7/15] - Train Loss: 0.5209, Val Loss: 1.9833, Train Acc: 90.9%, Val Acc: 80.9%, Val AUC: 0.696, Val P/R/F1: 0.50/0.18/0.26 - Elapsed: 00:14:30, Avg/Epoch: 00:02:03
16:52:22 • root: 
Epoch [8/15] - Train Loss: 0.4349, Val Loss: 1.9594, Train Acc: 93.9%, Val Acc: 79.1%, Val AUC: 0.729, Val P/R/F1: 0.43/0.30/0.35 - Elapsed: 00:16:34, Avg/Epoch: 00:02:03
16:54:26 • root: 
Epoch [9/15] - Train Loss: 0.3127, Val Loss: 1.8885, Train Acc: 95.8%, Val Acc: 77.3%, Val AUC: 0.712, Val P/R/F1: 0.40/0.39/0.40 - Elapsed: 00:18:38, Avg/Epoch: 00:02:03
16:56:29 • root: 
Epoch [10/15] - Train Loss: 0.1499, Val Loss: 2.3463, Train Acc: 98.2%, Val Acc: 80.2%, Val AUC: 0.652, Val P/R/F1: 0.47/0.24/0.31 - Elapsed: 00:20:41, Avg/Epoch: 00:02:03
16:58:33 • root: 
Epoch [11/15] - Train Loss: 0.1180, Val Loss: 2.2444, Train Acc: 98.6%, Val Acc: 80.2%, Val AUC: 0.669, Val P/R/F1: 0.47/0.30/0.36 - Elapsed: 00:22:45, Avg/Epoch: 00:02:03
17:00:37 • root: 
Epoch [12/15] - Train Loss: 0.0671, Val Loss: 2.2433, Train Acc: 99.3%, Val Acc: 80.2%, Val AUC: 0.673, Val P/R/F1: 0.47/0.30/0.36 - Elapsed: 00:24:49, Avg/Epoch: 00:02:03
17:02:41 • root: 
Epoch [13/15] - Train Loss: 0.0429, Val Loss: 2.1691, Train Acc: 99.6%, Val Acc: 77.7%, Val AUC: 0.696, Val P/R/F1: 0.41/0.37/0.39 - Elapsed: 00:26:53, Avg/Epoch: 00:02:03
17:04:44 • root: 
Epoch [14/15] - Train Loss: 0.0230, Val Loss: 2.3075, Train Acc: 99.7%, Val Acc: 77.7%, Val AUC: 0.666, Val P/R/F1: 0.41/0.37/0.39 - Elapsed: 00:28:55, Avg/Epoch: 00:02:03
17:06:48 • root: 
Epoch [15/15] - Train Loss: 0.0482, Val Loss: 2.3798, Train Acc: 99.5%, Val Acc: 78.4%, Val AUC: 0.700, Val P/R/F1: 0.42/0.37/0.39 - Elapsed: 00:31:00, Avg/Epoch: 00:02:03
17:06:48 • root: 

Training complete!
```

### wide_heads_deep

```json
"model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 160,
      "enc_nhead": 8,
      "enc_num_layers": 4,
      "agg_d_model": 160,
      "agg_nhead": 8,
      "agg_num_layers": 4,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  }
```

```bash
17:11:42 • root: 
Epoch [1/15] - Train Loss: 0.7570, Val Loss: 0.7403, Train Acc: 43.6%, Val Acc: 19.1%, Val AUC: 0.593, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:04:20, Avg/Epoch: 00:04:20 - New best! Model saved.
17:16:00 • root: 
Epoch [2/15] - Train Loss: 0.7699, Val Loss: 0.6878, Train Acc: 36.0%, Val Acc: 81.1%, Val AUC: 0.584, Val P/R/F1: 0.57/0.05/0.09 - Elapsed: 00:08:38, Avg/Epoch: 00:04:18
17:20:20 • root: 
Epoch [3/15] - Train Loss: 0.7585, Val Loss: 0.7310, Train Acc: 32.7%, Val Acc: 20.2%, Val AUC: 0.591, Val P/R/F1: 0.19/0.98/0.32 - Elapsed: 00:12:58, Avg/Epoch: 00:04:18
17:24:40 • root: 
Epoch [4/15] - Train Loss: 0.7238, Val Loss: 0.9801, Train Acc: 45.4%, Val Acc: 19.1%, Val AUC: 0.613, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:17:18, Avg/Epoch: 00:04:18 - New best! Model saved.
17:29:01 • root: 
Epoch [5/15] - Train Loss: 0.7567, Val Loss: 0.6890, Train Acc: 24.3%, Val Acc: 80.0%, Val AUC: 0.596, Val P/R/F1: 0.43/0.14/0.21 - Elapsed: 00:21:39, Avg/Epoch: 00:04:19
17:33:21 • root: 
Epoch [6/15] - Train Loss: 0.7624, Val Loss: 0.8966, Train Acc: 78.7%, Val Acc: 77.5%, Val AUC: 0.711, Val P/R/F1: 0.41/0.40/0.41 - Elapsed: 00:25:59, Avg/Epoch: 00:04:19 - New best! Model saved.
17:37:39 • root: 
Epoch [7/15] - Train Loss: 0.5866, Val Loss: 1.5490, Train Acc: 88.2%, Val Acc: 77.5%, Val AUC: 0.708, Val P/R/F1: 0.39/0.33/0.36 - Elapsed: 00:30:17, Avg/Epoch: 00:04:18
17:41:58 • root: 
Epoch [8/15] - Train Loss: 0.5070, Val Loss: 1.7827, Train Acc: 92.9%, Val Acc: 78.2%, Val AUC: 0.703, Val P/R/F1: 0.38/0.21/0.27 - Elapsed: 00:34:36, Avg/Epoch: 00:04:18
17:46:17 • root: 
Epoch [9/15] - Train Loss: 0.2913, Val Loss: 2.2804, Train Acc: 95.2%, Val Acc: 79.5%, Val AUC: 0.675, Val P/R/F1: 0.39/0.13/0.20 - Elapsed: 00:38:55, Avg/Epoch: 00:04:18
17:50:35 • root: 
Epoch [10/15] - Train Loss: 0.2452, Val Loss: 2.3086, Train Acc: 96.9%, Val Acc: 78.9%, Val AUC: 0.671, Val P/R/F1: 0.39/0.19/0.26 - Elapsed: 00:43:13, Avg/Epoch: 00:04:18
17:54:54 • root: 
Epoch [11/15] - Train Loss: 0.1515, Val Loss: 2.1852, Train Acc: 98.3%, Val Acc: 76.8%, Val AUC: 0.669, Val P/R/F1: 0.33/0.21/0.26 - Elapsed: 00:47:32, Avg/Epoch: 00:04:18
17:59:13 • root: 
Epoch [12/15] - Train Loss: 0.0908, Val Loss: 2.2300, Train Acc: 99.0%, Val Acc: 74.8%, Val AUC: 0.693, Val P/R/F1: 0.32/0.29/0.30 - Elapsed: 00:51:52, Avg/Epoch: 00:04:18
18:03:33 • root: 
Epoch [13/15] - Train Loss: 0.1044, Val Loss: 2.4673, Train Acc: 99.1%, Val Acc: 75.7%, Val AUC: 0.664, Val P/R/F1: 0.32/0.25/0.28 - Elapsed: 00:56:11, Avg/Epoch: 00:04:18
18:07:50 • root: 
Epoch [14/15] - Train Loss: 0.0881, Val Loss: 2.2921, Train Acc: 99.2%, Val Acc: 77.0%, Val AUC: 0.671, Val P/R/F1: 0.35/0.24/0.28 - Elapsed: 01:00:29, Avg/Epoch: 00:04:18
18:12:10 • root: 
Epoch [15/15] - Train Loss: 0.0826, Val Loss: 2.4837, Train Acc: 99.2%, Val Acc: 77.3%, Val AUC: 0.676, Val P/R/F1: 0.36/0.24/0.29 - Elapsed: 01:04:48, Avg/Epoch: 00:04:18
18:12:10 • root: 

Training complete!
```