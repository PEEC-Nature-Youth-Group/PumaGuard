"""
The Model class.
"""

import logging
import os
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Tuple,
)

import keras  # type: ignore
import tensorflow as tf  # type: ignore

from pumaguard.models import (
    __MODELS__,
)
from pumaguard.presets import (
    Preset,
)
from pumaguard.utils import (
    get_md5,
)

logger = logging.getLogger('PumaGuard')


class Model(ABC):
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

    @abstractmethod
    def raw_model(self, image_dimensions: Tuple[int, int]) -> keras.Model:
        """
        The model defined as layers (before it is compiled).
        """

    @property
    @abstractmethod
    def color_mode(self) -> str:
        """
        The color mode (rgb or grayscale) of the model.
        """

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

    def _compile_model(self, presets: Preset,
                       distribution_strategy: tf.distribute.Strategy) \
            -> keras.Model:
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
                if presets.model_function_name not in __MODELS__:
                    raise ValueError('unknown model function '
                                     f'{presets.model_function_name}')
                model = self.raw_model(presets.image_dimensions)

            logger.debug('Compiling model')
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=presets.alpha),
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
            logger.info('Number of layers in the model: %d', len(model.layers))
            model.summary()

        return model
