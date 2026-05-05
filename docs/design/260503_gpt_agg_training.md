# GPT-2 to Transformer Aggregator Model Training

The following training logs were recorded for various params:

## sm2

```json
{
  "model": {
    "module": "mean_agg",
    "class_name": "OscarNomAgg",
    "params": {
      "chunk_size": 1024,
      "enc_d_model": 768,
      "agg_d_model": 256,
      "agg_nhead": 8,
      "agg_num_layers": 4,
      "max_seq_len": 106578,
      "dropout": 0.2
    }
  },
  "training": {
    "epochs": 20,
    "batch_size": 8,
    "peak_lr": 0.0003,
    "weight_decay": 0.05,
    "eta_min": 1e-06,
    "warmup_fraction": 0.05,
    "grad_clip": 1.0,
    "class_weights": [
      1.0,
      8.0
    ],
    "label_smoothing": 0.1,
    "seed": 1337
  }
}
```

```bash
18:08:21 • root: 
Epoch [1/20] - Train Loss: 0.7737, Val Loss: 0.7936, Train Acc: 25.3%, Val Acc: 19.1%, Val AUC: 0.568, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:02, Avg/Epoch: 00:00:02 - New best (AUC/F1/LOSS)! Model saved.
18:08:22 • root: 
Epoch [2/20] - Train Loss: 0.7844, Val Loss: 0.7967, Train Acc: 20.0%, Val Acc: 77.3%, Val AUC: 0.497, Val P/R/F1: 0.19/0.06/0.09 - Elapsed: 00:00:04, Avg/Epoch: 00:00:01
18:08:24 • root: 
Epoch [3/20] - Train Loss: 0.7899, Val Loss: 0.7917, Train Acc: 19.8%, Val Acc: 19.1%, Val AUC: 0.530, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:06, Avg/Epoch: 00:00:01 - New best (LOSS)! Model saved.
18:08:26 • root: 
Epoch [4/20] - Train Loss: 0.7660, Val Loss: 0.7645, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.456, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:08, Avg/Epoch: 00:00:01 - New best (LOSS)! Model saved.
18:08:29 • root: 
Epoch [5/20] - Train Loss: 0.7760, Val Loss: 0.7606, Train Acc: 19.0%, Val Acc: 20.5%, Val AUC: 0.489, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:09, Avg/Epoch: 00:00:01 - New best (F1/LOSS)! Model saved.
18:08:31 • root: 
Epoch [6/20] - Train Loss: 0.7563, Val Loss: 0.7554, Train Acc: 25.7%, Val Acc: 25.5%, Val AUC: 0.609, Val P/R/F1: 0.20/0.96/0.33 - Elapsed: 00:00:12, Avg/Epoch: 00:00:01 - New best (AUC/F1/LOSS)! Model saved.
18:08:33 • root: 
Epoch [7/20] - Train Loss: 0.7430, Val Loss: 0.7357, Train Acc: 27.8%, Val Acc: 20.0%, Val AUC: 0.628, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:14, Avg/Epoch: 00:00:01 - New best (AUC/LOSS)! Model saved.
18:08:35 • root: 
Epoch [8/20] - Train Loss: 0.7517, Val Loss: 0.7388, Train Acc: 32.3%, Val Acc: 19.8%, Val AUC: 0.687, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:16, Avg/Epoch: 00:00:01 - New best (AUC)! Model saved.
18:08:37 • root: 
Epoch [9/20] - Train Loss: 0.7273, Val Loss: 0.8030, Train Acc: 45.3%, Val Acc: 69.5%, Val AUC: 0.654, Val P/R/F1: 0.32/0.52/0.40 - Elapsed: 00:00:18, Avg/Epoch: 00:00:01 - New best (F1)! Model saved.
18:08:39 • root: 
Epoch [10/20] - Train Loss: 0.7176, Val Loss: 0.7288, Train Acc: 40.2%, Val Acc: 26.8%, Val AUC: 0.679, Val P/R/F1: 0.20/0.95/0.33 - Elapsed: 00:00:20, Avg/Epoch: 00:00:01 - New best (LOSS)! Model saved.
18:08:43 • root: 
Epoch [11/20] - Train Loss: 0.7046, Val Loss: 0.7610, Train Acc: 46.1%, Val Acc: 49.5%, Val AUC: 0.706, Val P/R/F1: 0.25/0.82/0.38 - Elapsed: 00:00:24, Avg/Epoch: 00:00:01 - New best (AUC)! Model saved.
18:08:45 • root: 
Epoch [12/20] - Train Loss: 0.7020, Val Loss: 0.7605, Train Acc: 56.6%, Val Acc: 34.1%, Val AUC: 0.661, Val P/R/F1: 0.21/0.89/0.34 - Elapsed: 00:00:26, Avg/Epoch: 00:00:01
18:08:47 • root: 
Epoch [13/20] - Train Loss: 0.6760, Val Loss: 0.7647, Train Acc: 56.9%, Val Acc: 57.0%, Val AUC: 0.678, Val P/R/F1: 0.26/0.69/0.38 - Elapsed: 00:00:28, Avg/Epoch: 00:00:01
18:08:49 • root: 
Epoch [14/20] - Train Loss: 0.6725, Val Loss: 0.7809, Train Acc: 65.1%, Val Acc: 39.1%, Val AUC: 0.687, Val P/R/F1: 0.22/0.89/0.36 - Elapsed: 00:00:30, Avg/Epoch: 00:00:01
18:08:51 • root: 
Epoch [15/20] - Train Loss: 0.6523, Val Loss: 0.7490, Train Acc: 62.6%, Val Acc: 57.5%, Val AUC: 0.685, Val P/R/F1: 0.26/0.69/0.38 - Elapsed: 00:00:32, Avg/Epoch: 00:00:01
18:08:53 • root: 
Epoch [16/20] - Train Loss: 0.6449, Val Loss: 0.7713, Train Acc: 62.3%, Val Acc: 51.4%, Val AUC: 0.681, Val P/R/F1: 0.25/0.76/0.37 - Elapsed: 00:00:34, Avg/Epoch: 00:00:01
18:08:54 • root: 
Epoch [17/20] - Train Loss: 0.6511, Val Loss: 0.7884, Train Acc: 68.2%, Val Acc: 60.0%, Val AUC: 0.674, Val P/R/F1: 0.27/0.67/0.39 - Elapsed: 00:00:36, Avg/Epoch: 00:00:01
18:08:56 • root: 
Epoch [18/20] - Train Loss: 0.6376, Val Loss: 0.7858, Train Acc: 68.1%, Val Acc: 68.6%, Val AUC: 0.677, Val P/R/F1: 0.33/0.61/0.42 - Elapsed: 00:00:37, Avg/Epoch: 00:00:01 - New best (F1)! Model saved.
18:08:58 • root: 
Epoch [19/20] - Train Loss: 0.6312, Val Loss: 0.8089, Train Acc: 71.4%, Val Acc: 62.5%, Val AUC: 0.674, Val P/R/F1: 0.28/0.63/0.39 - Elapsed: 00:00:39, Avg/Epoch: 00:00:01
18:09:00 • root: 
Epoch [20/20] - Train Loss: 0.6147, Val Loss: 0.7761, Train Acc: 70.4%, Val Acc: 62.7%, Val AUC: 0.674, Val P/R/F1: 0.29/0.64/0.40 - Elapsed: 00:00:41, Avg/Epoch: 00:00:01
18:09:00 • root: 

Training complete!
18:09:00 • root: 
  Best AUC:  0.7062 (epoch 11) -> gpt_agg_sm2_best_auc.pth
18:09:00 • root: 
  Best F1:   0.4250 (epoch 18) -> gpt_agg_sm2_best_f1.pth
18:09:00 • root: 
  Min Loss:  0.7288 (epoch 10) -> gpt_agg_sm2_best_loss.pth
```

## med2

```json
{
  "model": {
    "module": "mean_agg",
    "class_name": "OscarNomAgg",
    "params": {
      "chunk_size": 1024,
      "enc_d_model": 768,
      "agg_d_model": 512,
      "agg_nhead": 8,
      "agg_num_layers": 6,
      "max_seq_len": 106578,
      "dropout": 0.3
    }
  },
  "training": {
    "epochs": 20,
    "batch_size": 4,
    "peak_lr": 0.00015,
    "weight_decay": 0.1,
    "eta_min": 1e-06,
    "warmup_fraction": 0.1,
    "grad_clip": 1.0,
    "class_weights": [
      1.0,
      8.0
    ],
    "label_smoothing": 0.1,
    "seed": 1337
  }
}
```

```bash
18:13:21 • root: 
Epoch [1/20] - Train Loss: 0.9064, Val Loss: 0.9048, Train Acc: 42.3%, Val Acc: 80.9%, Val AUC: 0.579, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:00:05, Avg/Epoch: 00:00:05 - New best (AUC/F1/LOSS)! Model saved.
18:13:28 • root: 
Epoch [2/20] - Train Loss: 0.8761, Val Loss: 0.8872, Train Acc: 33.3%, Val Acc: 19.1%, Val AUC: 0.538, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:16, Avg/Epoch: 00:00:05 - New best (F1/LOSS)! Model saved.
18:13:35 • root: 
Epoch [3/20] - Train Loss: 0.8846, Val Loss: 0.8290, Train Acc: 30.4%, Val Acc: 19.1%, Val AUC: 0.638, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:23, Avg/Epoch: 00:00:05 - New best (AUC/LOSS)! Model saved.
18:13:41 • root: 
Epoch [4/20] - Train Loss: 0.8689, Val Loss: 0.8433, Train Acc: 22.7%, Val Acc: 19.1%, Val AUC: 0.548, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:30, Avg/Epoch: 00:00:05
18:13:47 • root: 
Epoch [5/20] - Train Loss: 0.8606, Val Loss: 0.8666, Train Acc: 34.5%, Val Acc: 21.1%, Val AUC: 0.561, Val P/R/F1: 0.19/0.96/0.32 - Elapsed: 00:00:36, Avg/Epoch: 00:00:05
18:13:56 • root: 
Epoch [6/20] - Train Loss: 0.8868, Val Loss: 0.8559, Train Acc: 19.4%, Val Acc: 19.1%, Val AUC: 0.564, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:45, Avg/Epoch: 00:00:05
18:14:02 • root: 
Epoch [7/20] - Train Loss: 0.8789, Val Loss: 0.8460, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.574, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:51, Avg/Epoch: 00:00:05
18:14:07 • root: 
Epoch [8/20] - Train Loss: 0.8602, Val Loss: 0.8771, Train Acc: 29.8%, Val Acc: 19.1%, Val AUC: 0.533, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:00:57, Avg/Epoch: 00:00:05
18:14:13 • root: 
Epoch [9/20] - Train Loss: 0.8853, Val Loss: 0.8386, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.620, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:02, Avg/Epoch: 00:00:05
18:14:19 • root: 
Epoch [10/20] - Train Loss: 0.8622, Val Loss: 0.9922, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.570, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:08, Avg/Epoch: 00:00:05
18:14:28 • root: 
Epoch [11/20] - Train Loss: 0.8977, Val Loss: 0.9876, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.607, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:17, Avg/Epoch: 00:00:05
18:14:34 • root: 
Epoch [12/20] - Train Loss: 0.8800, Val Loss: 0.9926, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.579, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:23, Avg/Epoch: 00:00:05
18:14:41 • root: 
Epoch [13/20] - Train Loss: 0.9058, Val Loss: 0.8886, Train Acc: 20.7%, Val Acc: 19.1%, Val AUC: 0.536, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:30, Avg/Epoch: 00:00:05
18:14:48 • root: 
Epoch [14/20] - Train Loss: 0.8757, Val Loss: 0.9409, Train Acc: 26.1%, Val Acc: 19.1%, Val AUC: 0.459, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:37, Avg/Epoch: 00:00:05
18:14:54 • root: 
Epoch [15/20] - Train Loss: 0.9030, Val Loss: 0.9435, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.534, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:43, Avg/Epoch: 00:00:05
18:15:03 • root: 
Epoch [16/20] - Train Loss: 0.8950, Val Loss: 0.9776, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.487, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:52, Avg/Epoch: 00:00:05
18:15:10 • root: 
Epoch [17/20] - Train Loss: 0.8885, Val Loss: 0.9657, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.558, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:01:59, Avg/Epoch: 00:00:05
18:15:17 • root: 
Epoch [18/20] - Train Loss: 0.8879, Val Loss: 0.9208, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.516, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:02:06, Avg/Epoch: 00:00:05
18:15:23 • root: 
Epoch [19/20] - Train Loss: 0.9094, Val Loss: 0.9421, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.514, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:02:13, Avg/Epoch: 00:00:05
18:15:30 • root: 
Epoch [20/20] - Train Loss: 0.9032, Val Loss: 0.9351, Train Acc: 18.9%, Val Acc: 19.1%, Val AUC: 0.500, Val P/R/F1: 0.19/1.00/0.32 - Elapsed: 00:02:19, Avg/Epoch: 00:00:05
18:15:30 • root: 

Training complete!
18:15:30 • root: 
  Best AUC:  0.6384 (epoch 3) -> gpt_agg_med2_best_auc.pth
18:15:30 • root: 
  Best F1:   0.3206 (epoch 2) -> gpt_agg_med2_best_f1.pth
18:15:30 • root: 
  Min Loss:  0.8290 (epoch 3) -> gpt_agg_med2_best_loss.pth
```

## lg2

```json
{
  "model": {
    "module": "mean_agg",
    "class_name": "OscarNomAgg",
    "params": {
      "chunk_size": 1024,
      "enc_d_model": 768,
      "agg_d_model": 768,
      "agg_nhead": 12,
      "agg_num_layers": 12,
      "max_seq_len": 106578,
      "dropout": 0.4
    }
  },
  "training": {
    "epochs": 20,
    "batch_size": 2,
    "peak_lr": 5e-05,
    "weight_decay": 0.2,
    "eta_min": 1e-06,
    "warmup_fraction": 0.15,
    "grad_clip": 1.0,
    "class_weights": [
      1.0,
      8.0
    ],
    "label_smoothing": 0.1,
    "seed": 1337
  }
}
```

```bash
18:30:57 • root: 
W&B run initialized: id=15fuezbr, mode=offline, dir=../results/260503_gpt_sweep1/wandb/offline-run-20260502_183057-15fuezbr/files
18:31:43 • root: 
Epoch [1/20] - Train Loss: 0.9364, Val Loss: 0.9015, Train Acc: 65.2%, Val Acc: 80.9%, Val AUC: 0.554, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:00:30, Avg/Epoch: 00:00:30 - New best (AUC/F1/LOSS)! Model saved.
18:32:23 • root: 
Epoch [2/20] - Train Loss: 0.8925, Val Loss: 0.8986, Train Acc: 75.3%, Val Acc: 80.9%, Val AUC: 0.607, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:01:15, Avg/Epoch: 00:00:29 - New best (AUC/LOSS)! Model saved.
18:33:03 • root: 
Epoch [3/20] - Train Loss: 0.9040, Val Loss: 0.8742, Train Acc: 79.8%, Val Acc: 80.9%, Val AUC: 0.565, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:01:58, Avg/Epoch: 00:00:30 - New best (LOSS)! Model saved.
18:33:38 • root: 
Epoch [4/20] - Train Loss: 0.9024, Val Loss: 0.9488, Train Acc: 80.5%, Val Acc: 80.9%, Val AUC: 0.434, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:02:37, Avg/Epoch: 00:00:30
18:34:13 • root: 
Epoch [5/20] - Train Loss: 0.9024, Val Loss: 0.9835, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.534, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:03:12, Avg/Epoch: 00:00:30
18:34:46 • root: 
Epoch [6/20] - Train Loss: 0.9048, Val Loss: 0.9406, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.522, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:03:45, Avg/Epoch: 00:00:30
18:35:19 • root: 
Epoch [7/20] - Train Loss: 0.9123, Val Loss: 0.9304, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.519, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:04:18, Avg/Epoch: 00:00:30
18:35:55 • root: 
Epoch [8/20] - Train Loss: 0.9067, Val Loss: 0.9369, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.519, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:04:54, Avg/Epoch: 00:00:30
18:36:28 • root: 
Epoch [9/20] - Train Loss: 0.9038, Val Loss: 0.9099, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.431, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:05:27, Avg/Epoch: 00:00:30
18:37:01 • root: 
Epoch [10/20] - Train Loss: 0.9104, Val Loss: 0.9196, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.473, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:06:00, Avg/Epoch: 00:00:30
18:37:34 • root: 
Epoch [11/20] - Train Loss: 0.9112, Val Loss: 0.9619, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.547, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:06:33, Avg/Epoch: 00:00:30
18:38:08 • root: 
Epoch [12/20] - Train Loss: 0.9128, Val Loss: 0.9436, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.543, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:07:07, Avg/Epoch: 00:00:30
18:38:40 • root: 
Epoch [13/20] - Train Loss: 0.9118, Val Loss: 0.9319, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.527, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:07:39, Avg/Epoch: 00:00:30
18:39:13 • root: 
Epoch [14/20] - Train Loss: 0.9129, Val Loss: 0.9306, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.465, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:08:12, Avg/Epoch: 00:00:30
18:39:45 • root: 
Epoch [15/20] - Train Loss: 0.9118, Val Loss: 0.9547, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.521, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:08:44, Avg/Epoch: 00:00:30
18:40:19 • root: 
Epoch [16/20] - Train Loss: 0.9132, Val Loss: 0.9505, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.527, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:09:18, Avg/Epoch: 00:00:30
18:40:52 • root: 
Epoch [17/20] - Train Loss: 0.9119, Val Loss: 0.9435, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.514, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:09:51, Avg/Epoch: 00:00:30
18:41:25 • root: 
Epoch [18/20] - Train Loss: 0.9129, Val Loss: 0.9522, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.441, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:10:24, Avg/Epoch: 00:00:30
18:41:57 • root: 
Epoch [19/20] - Train Loss: 0.9103, Val Loss: 0.9412, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.508, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:10:56, Avg/Epoch: 00:00:30
18:42:32 • root: 
Epoch [20/20] - Train Loss: 0.9107, Val Loss: 0.9480, Train Acc: 81.1%, Val Acc: 80.9%, Val AUC: 0.569, Val P/R/F1: 0.00/0.00/0.00 - Elapsed: 00:11:30, Avg/Epoch: 00:00:30
18:42:32 • root: 

Training complete!
18:42:32 • root: 
  Best AUC:  0.6068 (epoch 2) -> gpt_agg_lg2_best_auc.pth
18:42:32 • root: 
  Best F1:   0.0000 (epoch 1) -> gpt_agg_lg2_best_f1.pth
18:42:32 • root: 
  Min Loss:  0.8742 (epoch 3) -> gpt_agg_lg2_best_loss.pth
```