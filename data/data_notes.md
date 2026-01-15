# Dataset notes

HuggingFace Dataset: https://huggingface.co/datasets/Francis2003/Movie-O-Label

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

Mean-min-max script length in terms of number of tokens

Tokens (GPT-2 tokenizer):
```
train dataset: mean=37082.5, min=7008, max=106,578
val dataset: mean=37251.7, min=13063, max=72,759
test dataset: mean=36871.5, min=11097, max=94,792
global dataset: mean=37074.1, min=7008, max=106,578
```

Mean-min-max summary length in terms of number of tokens

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