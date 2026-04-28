#!/bin/bash

# Run training first
python train.py --config config_mean.json

# Run testing after training completes
python test.py --config config_mean.json