"""
Some utility functions.
"""

import tensorflow as tf  # type: ignore


def initialize_tensorflow() -> tf.distribute.Strategy:
    """
    Initialize Tensorflow on available hardware.

    Try different backends in the following order: TPU, GPU, CPU and use the
    first one available.

    Returns:
        tf.distribute.Strategy: The distribution strategy object after
        initialization.
    """
    print("Tensorflow version " + tf.__version__)
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
        return tf.distribute.TPUStrategy(tpu)
    except ValueError:
        print("WARNING: Not connected to a TPU runtime; Will try GPU")
        if tf.config.list_physical_devices('GPU'):
            print('Running on '
                  f'{len(tf.config.list_physical_devices("GPU"))} GPUs')
            return tf.distribute.MirroredStrategy()
        print('WARNING: Not connected to TPU or GPU runtime; '
              'Will use CPU context')
        return tf.distribute.get_strategy()
