# Mean-Pool Transformer Model Hyperparameter Sweep

"Story view": Sweep starting with smallest then progressively changing only one parameter and seeing impact
| Name | AUC | Acc | F1 | Macro F1 | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L |
|---|---|---|---|---|---|---|---|---|---|---|
| start_test | 0.7334 | 0.8136 | 0.2115 | 0.5529 | 96 | 4 | 2 | 96 | 4 | 2 |
| 02_test | 0.7179 | 0.7205 | 0.3941 | 0.6062 | 96 | 4 | 3 | 96 | 4 | 2 |
| 03_test | 0.7297 | 0.7636 | 0.3810 | 0.6174 | 96 | 4 | 3 | 96 | 4 | 3 |
| 04_test | 0.7172 | 0.7341 | 0.3607 | 0.5964 | 96 | 4 | 4 | 96 | 4 | 3 |
| 05_test | 0.7171 | 0.7773 | 0.3718 | 0.6182 | 96 | 4 | 4 | 96 | 4 | 4 |
| 06_test | 0.7328 | 0.8136 | 0.2407 | 0.5673 | 96 | 6 | 2 | 96 | 4 | 2 |
| 07_test | 0.7253 | 0.8091 | 0.2075 | 0.5495 | 96 | 6 | 2 | 96 | 6 | 2 |
| 08_test | 0.7208 | 0.6545 | 0.4370 | 0.5939 | 96 | 6 | 3 | 96 | 6 | 2 |
| 09_test | 0.7261 | 0.8091 | 0.0455 | 0.4697 | 96 | 6 | 3 | 96 | 6 | 3 |
| 10_test | 0.7138 | 0.7409 | 0.3736 | 0.6052 | 96 | 6 | 4 | 96 | 6 | 3 |
| 11_test | 0.7099 | 0.7659 | 0.3602 | 0.6085 | 96 | 6 | 4 | 96 | 6 | 4 |
| 12_test | 0.7220 | 0.8091 | 0.2364 | 0.5636 | 128 | 4 | 2 | 128 | 4 | 2 |
| 13_test | 0.7292 | 0.8136 | 0.2679 | 0.5805 | 128 | 8 | 2 | 128 | 4 | 2 |
| 14_test | 0.7213 | 0.6477 | 0.3969 | 0.5740 | 160 | 8 | 2 | 128 | 4 | 2 |

Other tests focusing on changes to training parameters:

| Name | AUC | Acc | F1 | Macro F1 | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L | dropout | cls_w | peak_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 260430_start_test | 0.7342 | 0.8114 | 0.2655 | 0.5786 | 128 | 4 | 2 | 128 | 4 | 2 | 0.4 | [1,8] | 1e-4 |
| 260502_12_test | 0.7220 | 0.8091 | 0.2364 | 0.5636 | 128 | 4 | 2 | 128 | 4 | 2 | 0.5 | [1,4] | 5e-5 |
| 260430_narrow_test | 0.7268 | 0.8068 | 0.1414 | 0.5163 | 96 | 4 | 2 | 96 | 4 | 2 | 0.4 | [1,8] | 1e-4 |
| 260502_start_test | 0.7334 | 0.8136 | 0.2115 | 0.5529 | 96 | 4 | 2 | 96 | 4 | 2 | 0.5 | [1,4] | 5e-5 |

"Rank view", best in each metric so far:

| Name | AUC | Acc | F1 | Macro F1 | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L |
|---|---|---|---|---|---|---|---|---|---|---|
| best_auc(260430_start) | 0.7342 | 0.8114 | 0.2655 | 0.5786 | 128 | 4 | 2 | 128 | 4 | 2 |
| best_acc2(13) | 0.7292 | 0.8136 | 0.2679 | 0.5805 | 128 | 8 | 2 | 128 | 4 | 2 |
| best_acc1(260502_start) | 0.7334 | 0.8136 | 0.2115 | 0.5529 | 96 | 4 | 2 | 96 | 4 | 2 |
| best_macrof1(5) | 0.7171 | 0.7773 | 0.3718 | 0.6182 | 96 | 4 | 4 | 96 | 4 | 4 |
| best_acc2(6) | 0.7328 | 0.8136 | 0.2407 | 0.5673 | 96 | 6 | 2 | 96 | 4 | 2 |
| best_f1(8) | 0.7208 | 0.6545 | 0.4370 | 0.5939 | 96 | 6 | 3 | 96 | 6 | 2 |

## Next sweep

| Name | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L |
|---|---|---|---|---|---|---|
| 01 | 128 | 4 | 2 | 128 | 4 | 2 |
| 02 | 128 | 8 | 2 | 128 | 4 | 2 |
| 03 | 96 | 4 | 2 | 96 | 4 | 2 |
| 04 | 96 | 4 | 4 | 96 | 4 | 4 |
| 05 | 96 | 6 | 2 | 96 | 4 | 2 |
| 06 | 96 | 6 | 3 | 96 | 6 | 2 |
| 07 | 96 | 8 | 2 | 96 | 4 | 2 |
| 08 | 96 | 8 | 3 | 96 | 4 | 2 |


For 128-d models: Use this "aggresive" training config (note dropout as well)

```json
{
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
      "dropout": 0.5
    }
  },
  "training": {
    "epochs": 15,
    "batch_size": 4,
    "peak_lr": 1e-4,
    "weight_decay": 0.2,
    "eta_min": 1e-06,
    "warmup_fraction": 0.1,
    "grad_clip": 1.0,
    "class_weights": [
      1.0,
      10.0
    ],
    "label_smoothing": 0.05,
    "seed": 1337
  }
}
```

For 96-d models: Use this "gentler" training config

```json
{
  "model": {
    "module": "mean_transformer",
    "class_name": "OscarNomTransformer",
    "params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": 96,
      "enc_nhead": 4,
      "enc_num_layers": 2,
      "agg_d_model": 96,
      "agg_nhead": 4,
      "agg_num_layers": 2,
      "max_seq_len": 106578,
      "dropout": 0.5
    }
  },
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
    "label_smoothing": 0.05,
    "seed": 1337
  }
}
```