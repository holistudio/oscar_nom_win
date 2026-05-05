#!/bin/bash

# Run training first
python emb_train.py --config emb_config.json

# Run testing after training completes
python emb_test.py --config emb_config.json