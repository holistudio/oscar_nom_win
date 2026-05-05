# Last Runs

Last runs took the best transformer modles from previous sweep and tweaked training parameters.

For enc_d = 128: More "aggressive"

```json
  "training": {
    "epochs": 15,
    "batch_size": 4,
    "peak_lr": 0.0001,
    "weight_decay": 0.2,
    "eta_min": 1e-06,
    "warmup_fraction": 0.1,
    "grad_clip": 1.0,
    "class_weights": [
      1.0,
      10.0
    ],
    "label_smoothing": 0.05
  }
```

For enc_d = 96: "gentler"

For enc_d = 128: More "aggressive"

```json
  "training": {
    "epochs": 15,
    "batch_size": 4,
    "peak_lr": 3e-05,
    "weight_decay": 0.15,
    "eta_min": 1e-06,
    "warmup_fraction": 0.15,
    "grad_clip": 1.0,
    "class_weights": [
      1.0,
      3.0
    ],
    "label_smoothing": 0.05
  }
```

Threshold tuning is also introduced using the validation dataset after every epoch and saving it if it is at least one of the three best checkpoints (loss/AUC/F1)

Slight tweaks were made to warmup fraction, weight decay and label smoothing. See `260505_latest_view.csv` for best results and corresponding parameters for each class of models "gpt_agg", "mean_trf", and "causal_trf"