# Causal vs Mean-Pool Transformer Model Selection

The following results were obtained after training `causal_transformer` and `mean_transformer` and evaluating on the test dataset

Sweeps with both types of transformers were trained with the same ranges of parameters:

```json
"params": {
      "chunk_size": 1024,
      "vocab_size": 50257,
      "enc_d_model": [96, 128, 160],
      "enc_nhead": [2, 4, 8],
      "enc_num_layers": [1, 2, 4],
      "agg_d_model": [96, 128, 160],
      "agg_nhead": [2, 4, 8],
      "agg_num_layers": [1, 2, 4],
      "max_seq_len": 106578,
      "dropout": 0.4
    }
```

Encoder and Aggregator always had the same parameters.

Below are results for `causal_transformer`:

| module | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L | acc | auc | prec | rec | f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| causal | 128 | 4 | 2 | 128 | 4 | 2 | 0.8114 | 0.6940 | 0 | 0 | 0 |
| causal | 128 | 2 | 1 | 128 | 2 | 1 | 0.8114 | 0.6780 | 0 | 0 | 0 |
| causal | 128 | 4 | 1 | 128 | 4 | 1 | 0.8114 | 0.6763 | 0 | 0 | 0 |
| causal | 96 | 2 | 1 | 96 | 2 | 1 | 0.8114 | 0.6580 | 0 | 0 | 0 |
| causal | 160 | 4 | 2 | 160 | 4 | 2 | 0.8114 | 0.6552 | 0 | 0 | 0 |
| causal | 96 | 4 | 1 | 96 | 4 | 1 | 0.8114 | 0.6212 | 0 | 0 | 0 |
| causal | 128 | 2 | 2 | 128 | 2 | 2 | 0.8023 | 0.6786 | 0.300 | 0.0361 | 0.0645 |
| causal | 96 | 4 | 2 | 96 | 4 | 2 | 0.7977 | 0.6371 | 0.400 | 0.1446 | 0.2124 |
| causal | 128 | 8 | 4 | 128 | 8 | 4 | 0.7909 | 0.7128 | 0.3548 | 0.1325 | 0.1930 |
| causal | 128 | 8 | 2 | 128 | 8 | 2 | 0.7659 | 0.6926 | 0.3837 | 0.3976 | 0.3905 |
| causal | 96 | 2 | 2 | 96 | 2 | 2 | 0.7500 | 0.6730 | 0.3579 | 0.4096 | 0.3820 |
| causal | 160 | 8 | 2 | 160 | 8 | 2 | 0.7000 | 0.6484 | 0.3185 | 0.5181 | 0.3945 |
| causal | 128 | 4 | 4 | 128 | 4 | 4 | 0.6114 | 0.7119 | 0.2822 | 0.6867 | 0.4000 |
| causal | 160 | 8 | 4 | 160 | 8 | 4 | 0.5000 | 0.6094 | 0.2355 | 0.7349 | 0.3567 |
| causal | 160 | 4 | 4 | 160 | 4 | 4 | 0.1886 | 0.6652 | 0.1886 | 1.0000 | 0.3174 |


Below are results for `mean_transformer`:

| module | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L | acc | auc | prec | rec | f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 128 | 4 | 2 | 128 | 4 | 2 | 0.8114 | 0.7342 | 0.500 | 0.1807 | 0.2655 |
| mean | 128 | 4 | 4 | 128 | 4 | 4 | 0.8114 | 0.6635 | 0 | 0 | 0 |
| mean | 128 | 2 | 1 | 128 | 2 | 1 | 0.8091 | 0.7316 | 0.4667 | 0.0843 | 0.1429 |
| mean | 128 | 2 | 2 | 128 | 2 | 2 | 0.8091 | 0.6820 | 0.480 | 0.1446 | 0.2222 |
| mean | 96 | 4 | 2 | 96 | 4 | 2 | 0.8068 | 0.7268 | 0.4375 | 0.0843 | 0.1414 |
| mean | 96 | 2 | 2 | 96 | 2 | 2 | 0.8068 | 0.7143 | 0.250 | 0.0120 | 0.0230 |
| mean | 128 | 8 | 2 | 128 | 8 | 2 | 0.7977 | 0.7196 | 0.200 | 0.0241 | 0.0430 |
| mean | 160 | 4 | 4 | 160 | 4 | 4 | 0.7886 | 0.7382 | 0.4306 | 0.3735 | 0.4000 |
| mean | 160 | 4 | 2 | 160 | 4 | 2 | 0.7886 | 0.7302 | 0.4194 | 0.3133 | 0.3586 |
| mean | 128 | 8 | 4 | 128 | 8 | 4 | 0.7568 | 0.7209 | 0.3421 | 0.3133 | 0.3270 |
| mean | 160 | 8 | 4 | 160 | 8 | 4 | 0.7477 | 0.7335 | 0.3571 | 0.4217 | 0.3867 |
| mean | 96 | 4 | 1 | 96 | 4 | 1 | 0.7386 | 0.7204 | 0.340 | 0.4096 | 0.3716 |
| mean | 96 | 2 | 1 | 96 | 2 | 1 | 0.7227 | 0.7196 | 0.3211 | 0.4217 | 0.3646 |
| mean | 160 | 8 | 2 | 160 | 8 | 2 | 0.7091 | 0.7236 | 0.3427 | 0.5904 | 0.4336 |
| mean | 128 | 4 | 1 | 128 | 4 | 1 | 0.7068 | 0.7355 | 0.3380 | 0.5783 | 0.4267 |


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

## Key takeaways

- The test dataset has 18.86% of screenplays nominated for Oscar, and therefore 81.14% of screenplays not nominated
- None of the models achieved an accuracy higher than 81.14% so none did better than a "dumb model" that only predicted "not nominated" for every screenplay in the test dataset.
- Overall `mean_transformer` did better than their `causal transformer` equivalents. So focus on `mean_transformer`

| module | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L | acc | auc | prec | rec | f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 128 | 4 | 2 | 128 | 4 | 2 | 0.8114 | 0.7342 | 0.500 | 0.1807 | 0.2655 |
| mean | 128 | 4 | 4 | 128 | 4 | 4 | 0.8114 | 0.6635 | 0 | 0 | 0 |
| mean | 128 | 2 | 1 | 128 | 2 | 1 | 0.8091 | 0.7316 | 0.4667 | 0.0843 | 0.1429 |
| mean | 128 | 2 | 2 | 128 | 2 | 2 | 0.8091 | 0.6820 | 0.480 | 0.1446 | 0.2222 |
| mean | 128 | 8 | 2 | 128 | 8 | 2 | 0.7977 | 0.7196 | 0.200 | 0.0241 | 0.0430 |
| mean | 128 | 8 | 4 | 128 | 8 | 4 | 0.7568 | 0.7209 | 0.3421 | 0.3133 | 0.3270 |
| mean | 128 | 4 | 1 | 128 | 4 | 1 | 0.7068 | 0.7355 | 0.3380 | 0.5783 | 0.4267 |

## Parameters that warrant further testing

- Of the models, `(d_model, nheads, layers) = [128, 4, 2]` did "the best" because it achieved the 81.14% accuracy and the highest AUC=0.734
- Among models where `d_model=128`
  - Increasing `layers` doesn't seem to help when you keep `nheads` the same
  - Increasing `nheads` from 4 to 8 doesn't seem to help
  - In general, the performance ceiling has been hit IF you keep parameters the same for encoder and aggregator.
- So the only thing worth trying with `d_model=128` is to explore "asymmetric" parameters. The key idea being that the encode has to be bigger since it is dealing with raw tokens in a larger dimensional feature space, while the aggregator only needs to deal with a shorter sequence of "compressed embeddings"
  - Increase `enc_nheads` to 8 but keep `agg_nheads` to 4 

| module | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L |
|---|---|---|---|---|---|---|
| mean | 128 | 8 | 2 | 128 | 4 | 2 |
| mean | 160 | 8 | 2 | 128 | 4 | 2 |

- Trends are also noticeable when focusing on models with `d_model=96`

 
| module | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L | acc | auc | prec | rec | f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 96 | 4 | 2 | 96 | 4 | 2 | 0.8068 | 0.7268 | 0.4375 | 0.0843 | 0.1414 |
| mean | 96 | 2 | 2 | 96 | 2 | 2 | 0.8068 | 0.7143 | 0.250 | 0.0120 | 0.0230 |
| mean | 96 | 4 | 1 | 96 | 4 | 1 | 0.7386 | 0.7204 | 0.340 | 0.4096 | 0.3716 |
| mean | 96 | 2 | 1 | 96 | 2 | 1 | 0.7227 | 0.7196 | 0.3211 | 0.4217 | 0.3646 |

- More layers and more heads helps
  - But up to a point, especially with num heads (see `d_model=128` table before)
- Along these ideas, asymmetric parameters are worth tryging as well

| module | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L |
|---|---|---|---|---|---|---|
| mean | 96 | 4 | 3 | 96 | 4 | 2 |
| mean | 96 | 4 | 3 | 96 | 4 | 3 |
| mean | 96 | 4 | 4 | 96 | 4 | 3 |
| mean | 96 | 4 | 4 | 96 | 4 | 4 |
| mean | 96 | 6 | 2 | 96 | 4 | 2 |
| mean | 96 | 6 | 2 | 96 | 6 | 2 |
| mean | 96 | 6 | 3 | 96 | 6 | 2 |
| mean | 96 | 6 | 3 | 96 | 6 | 3 |
| mean | 96 | 6 | 4 | 96 | 6 | 4 |

If training parameters are changed, "original runs" need to be re-run as well. So the next full sweep should be:

| module | enc_d | enc_h | enc_L | agg_d | agg_h | agg_L |
|---|---|---|---|---|---|---|
| mean | 96 | 4 | 2 | 96 | 4 | 2 |
| mean | 96 | 4 | 3 | 96 | 4 | 2 |
| mean | 96 | 4 | 3 | 96 | 4 | 3 |
| mean | 96 | 4 | 4 | 96 | 4 | 3 |
| mean | 96 | 4 | 4 | 96 | 4 | 4 |
| mean | 96 | 6 | 2 | 96 | 4 | 2 |
| mean | 96 | 6 | 2 | 96 | 6 | 2 |
| mean | 96 | 6 | 3 | 96 | 6 | 2 |
| mean | 96 | 6 | 3 | 96 | 6 | 3 |
| mean | 96 | 6 | 4 | 96 | 6 | 3 |
| mean | 96 | 6 | 4 | 96 | 6 | 4 |
| mean | 128 | 4 | 2 | 128 | 4 | 2 |
| mean | 128 | 8 | 2 | 128 | 4 | 2 |
| mean | 160 | 8 | 2 | 128 | 4 | 2 |