# after running first time, load files with uncommented code below

import os
import json
import tensorflow as tf

from gpt_download import load_gpt2_params_from_tf_ckpt

model_size="124M"
models_dir="gpt2"

# Define paths
model_dir = os.path.join(models_dir, model_size)

# Load settings and params
tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)