# Ideas

## Data

https://huggingface.co/datasets/Francis2003/Movie-O-Label

Metadata
- Title `title`
- Year `year`
- Movie Name = `title_year` format
- IMDB ID `imdb_id`
- Summary (almost a primary text input in and of itself...)

Scripts come in three formats:
- `script` has XML tags `<character>` `<scene>`
- `script_plain` has XML tags removed
- `script_clean` is plain with a few other processing: unicode normaliziation, stage directions and scene transitions stripped where possible, whitespace reduced

Optionally embeddings can be used:
- Script embeddings
- Summary embeddings
- Title embeddings

> The best-performing baseline (logistic regression) used **script_clean + summary + title** embeddings
> and achieved **ROC-AUC ≈ 0.79** and **Macro-F1 ≈ 0.68** on the test set.

Mean-min-max script length in terms of number of words and number of tokens

Tokens (GPT-2 tokenizer):
```
train dataset: mean=37082.5, min=7008, max=106578
val dataset: mean=37251.7, min=13063, max=72759
test dataset: mean=36871.5, min=11097, max=94792
global dataset: mean=37074.1, min=7008, max=106578
```

Mean-min-max summary length in terms of number of words and number of tokens

Tokens (GPT-2 tokenizer):
```
train dataset: mean=789.8, min=14, max=2282
val dataset: mean=795.3, min=33, max=1691
test dataset: mean=789.8, min=27, max=1723
global dataset: mean=790.9, min=14, max=2282
```

Training-validation-test samples:
`(1320, 440, 440)`

Imbalance percent nominations/wins

```
train: 18.94% nominated for Oscar in best screenplay
val: 19.09% nominated for Oscar in best screenplay
test: 18.86% nominated for Oscar in best screenplay
```

```
train: 4.39% won Oscar for best screenplay
val: 4.09% won Oscar for best screenplay
test: 5.00% won Oscar for best screenplay
```

## Neural network architectures and approaches

### Natural Language

#### Word counts
1. Identify the top 50 most frequently used words for each movie script (excluding articles and other "non-content" words)
2. Do this for each script in the training set and aggregate the words in a dictionary
3. Visualize distribution of these words for movies that did and didn't get nominated.
4. Find some heuristic that seems to work well enough in the training set.
5. See how good that is in the validation and test dataset.

The "cool" part of the above is ending up with something interpretable like "oh this script has these words that frequently come up

### RNN

Encoder Decoder LSTM

### Transformer

### GPT-2

Transformer Encoder and un-trained GPT-2

Transformer Encoder and pre-trained GPT-2

~~GPT-2's sub-modules but re-configured...eh no. OK you can change the output size easily - just add a binary classifciatin head/linear layer that re-maps the 50k vocab size to a single logit, BUT ALSO you need to increase the input size from 1600 to 106,578 and that's going to result in a memory issue for the attention matrix in the Transformer Blocks~~ 

Only look at the first 1024 tokens of each summary, predict Oscar nomination using pre-trained GPT-2 + binary classifier head.

### Open Source Local LLMs

System message: "Read the following movie script and predict its probability of getting nominated for an Academy Award/Oscar in any category. Respond only with a decimal value ranging between 0 and 1, inclusive."

User message: an entire script

Measure its classification performance out-of-the-box.

Figure out a way to fine tune it.
## Class imbalance methods