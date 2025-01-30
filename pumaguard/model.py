"""
The Model class.
"""

import logging
import os

import keras  # type: ignore
import tensorflow as tf  # type: ignore

from pumaguard.presets import (
    Preset,
)
from pumaguard.utils import (
    get_md5,
)

logger = logging.getLogger('PumaGuard')


class Model():
    """
    The Model used.
    """

    _instance = None
    _initialized = False

    def __new__(cls, presets: Preset):
        """
        Create a new model.
        """
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
        return cls._instance

    def __init__(self, presets: Preset):
        if not self._initialized:
            self._presets = presets
            self._distribution_strategy = self.initialize_tensorflow()
            self._model = self.create_model(
                self._presets, self._distribution_strategy)
            self._initialized = True

    def get_model(self) -> keras.src.Model:
        """
        Get the model.
        """
        return self._model

    def initialize_tensorflow(self) -> tf.distribute.Strategy:
        """
        Initialize Tensorflow on available hardware.

        Try different backends in the following order: TPU, GPU, CPU and use
        the first one available.

        Returns:
            tf.distribute.Strategy: The distribution strategy object after
            initialization.
        """
        logger.info("Tensorflow version %s", tf.__version__)
        logger.info('Trying to connect to a TPU')
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            logger.info(
                'Running on a TPU with %d cores',
                tpu.num_accelerators()["TPU"])
            return tf.distribute.TPUStrategy(tpu)
        except ValueError:
            logger.warning(
                "WARNING: Not connected to a TPU runtime; Will try GPU")
            if tf.config.list_physical_devices('GPU'):
                logger.info('Running on %d GPUs', len(
                    tf.config.list_physical_devices("GPU")))
                return tf.distribute.MirroredStrategy()
            logger.warning('WARNING: Not connected to TPU or GPU runtime; '
                           'Will use CPU context')
            return tf.distribute.get_strategy()

    def create_model(self, presets: Preset,
                     distribution_strategy: tf.distribute.Strategy) \
            -> keras.src.Model:
        """
        Create the model.
        """
        with distribution_strategy.scope():
            logger.info('looking for model at %s', presets.model_file)
            model_file_exists = os.path.isfile(presets.model_file)
            if presets.load_model_from_file and model_file_exists:
                os.stat(presets.model_file)
                logger.debug('loading model from file %s', presets.model_file)
                model = keras.models.load_model(presets.model_file)
                logger.debug('loaded model from file')
                logger.info('model version %s', get_md5(presets.model_file))
            else:
                if not presets.load_model_from_file:
                    logger.info('not loading previous weights')
                else:
                    logger.info('could not find model; creating new model')
                logger.debug('creating new %s model',
                             presets.model_function_name)
                model = presets.model_function(presets.image_dimensions)

            logger.debug('Compiling model')
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=presets.alpha),
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
            logger.info('Number of layers in the model: %d', len(model.layers))
            model.summary()

        return model
