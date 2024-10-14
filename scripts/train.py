"""
Train a model.
"""

import os
import tensorflow as tf

# Use data from local directory
base_data_directory = os.path.realpath('../data')
base_output_directory = os.path.realpath('../models')

# Initialize Tensorflow

# Imports tensorflow into notebook, which has the Xception model defined inside
# it
print("Tensorflow version " + tf.__version__)

# Try different backends in the following order: TPU, GPU, CPU and use the
# first one available
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    distribution_strategy = tf.distribute.TPUStrategy(tpu)
    print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
except ValueError:
    print("WARNING: Not connected to a TPU runtime; Will try GPU")
    if tf.config.list_physical_devices('GPU'):
        distribution_strategy = tf.distribute.MirroredStrategy()
        print(f'Running on {len(tf.config.list_physical_devices("GPU"))} GPUs')
    else:
        print('WARNING: Not connected to TPU or GPU runtime; '
              'Will use CPU context')
        distribution_strategy = tf.distribute.get_strategy()
