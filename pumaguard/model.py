"""
The Model class.
"""

import logging
import os
from typing import (
    Any,
)

import keras  # type: ignore
import tensorflow as tf  # type: ignore

from pumaguard.models import (
    __MODEL_FUNCTIONS__,
)
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

    _instance: Any = None
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
            self._distribution_strategy = self._initialize_tensorflow()
            self._model = self._compile_model(
                self._presets, self._distribution_strategy)
            self._initialized = True

    @property
    def model(self) -> keras.Model:
        """
        Get the model.
        """
        return self._model

    def _initialize_tensorflow(self) -> tf.distribute.Strategy:
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

    def _load_model(self, filename: str) -> keras.Model:
        """
        Load a model from file.
        """
        os.stat(filename)
        logger.debug('loading model from file %s', filename)
        model = keras.models.load_model(filename)
        logger.debug('loaded model from file')
        logger.info('loaded model version %s', get_md5(filename))
        return model

    def _compile_model(self, presets: Preset,
                       distribution_strategy: tf.distribute.Strategy) \
            -> keras.Model:
        """
        Create the model.
        """
        with distribution_strategy.scope():
            if presets.load_model_from_file:
                logger.info('looking for model at %s', presets.model_file)
                model_file_exists = os.path.isfile(presets.model_file)
                if model_file_exists:
                    model = self._load_model(presets.model_file)
                else:
                    raise FileNotFoundError(
                        f'could not find model {presets.model_file}')
            else:
                logger.debug('not loading previous weights')
                logger.info('creating new %s model',
                            presets.model_function_name)
                if presets.model_function_name not in __MODEL_FUNCTIONS__:
                    raise ValueError('unknown model function '
                                     f'{presets.model_function_name}')
                model = __MODEL_FUNCTIONS__[presets.model_function_name](
                    presets.image_dimensions)

            logger.debug('Compiling model')
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=presets.alpha),
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
            logger.info('Number of layers in the model: %d', len(model.layers))
            model.summary()

        return model
