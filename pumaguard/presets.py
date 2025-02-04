"""
The presets for each model.
"""

import copy
import logging
import os
from typing import (
    Tuple,
)

import tensorflow as tf  # type: ignore
import yaml
from packaging import (
    version,
)

from pumaguard.models import (
    __MODEL_FUNCTIONS__,
)

logger = logging.getLogger('PumaGuard')


# pylint: disable=too-many-public-methods
class Preset():
    """
    Base class for Presets
    """

    _alpha: float = 0
    _base_data_directory: str = ''
    _base_output_directory: str = ''

    def __init__(self):
        self.alpha = 1e-5
        self.base_data_directory = os.path.join(
            os.path.dirname(__file__), '../data')
        self.base_output_directory = os.path.join(
            os.path.dirname(__file__), '../models')
        self.batch_size = 16
        self.color_mode = 'rgb'
        self.epochs = 300
        self.image_dimensions: tuple[int, int] = (128, 128)
        self.lion_directories: list[str] = []
        self.load_history_from_file = True
        self.load_model_from_file = True
        self.model_function_name = 'pretrained'
        self.model_version = 'undefined'
        self.no_lion_directories: list[str] = []
        self.with_augmentation = False
        if version.parse(tf.__version__) < version.parse('2.17'):
            self.tf_compat = '2.15'
        else:
            self.tf_compat = '2.17'

    def load(self, filename: str):
        """
        Load settings from YAML file.
        """
        logger.info('loading settings from %s', filename)
        with open(filename, encoding='utf-8') as fd:
            settings = yaml.safe_load(fd)
        self.notebook_number = settings.get('notebook', 1)
        self.epochs = settings.get('epochs', 1)
        dimensions = settings.get('image-dimensions', [0, 0])
        if not isinstance(dimensions, list) or \
            len(dimensions) != 2 or \
                not all(isinstance(d, int) for d in dimensions):
            raise ValueError(
                'expected image-dimensions to be a list of two integers')
        self.image_dimensions = tuple(dimensions)
        self.model_version = settings.get('model-version', 'undefined')
        self.model_function_name = settings.get(
            'model-function', 'undefined')
        self.base_data_directory = settings.get(
            'base-data-directory', 'undefined')
        self.base_output_directory = settings.get(
            'base-output-directory', 'undefined')
        lions = settings.get('lion-directories', ['undefined'])
        if not isinstance(lions, list) or \
                not all(isinstance(p, str) for p in lions):
            raise ValueError('expected lion-directories to be a list of paths')
        self.lion_directories = lions
        no_lions = settings.get('no-lion-directories', ['undefined'])
        if not isinstance(no_lions, list) or \
                not all(isinstance(p, str) for p in no_lions):
            raise ValueError(
                'expected no-lion-directories to be a list of paths')
        self.no_lion_directories = no_lions
        self.with_augmentation = settings.get('with-augmentation', False)
        self.batch_size = settings.get('batch-size', 1)
        self.alpha = float(settings.get('alpha', 1e-5))
        self.color_mode = settings.get('color-mode', 'rgb')

    def save(self):
        """
        Write presets to standard output.
        """
        yaml.dump(self)

    def _relative_paths(self, base: str, paths: list[str]) -> list[str]:
        """
        The directories relative to a base path.
        """
        return [
            os.path.relpath(path, start=base)for path in paths
        ]

    def __iter__(self):
        """
        Serialize this class.
        """
        yield from {
            'alpha': self.alpha,
            'base-data-directory': os.path.relpath(
                self.base_data_directory,
                os.path.dirname(__file__)),
            'batch-size': self.batch_size,
            'color-mode': self.color_mode,
            'epochs': self.epochs,
            'image-dimensions': self.image_dimensions,
            'lion-directories': self._relative_paths(
                self.base_data_directory,
                self.lion_directories),
            'model-function': self.model_function_name,
            'model-version': self.model_version,
            'no-lion-directories': self._relative_paths(
                self.base_data_directory,
                self.no_lion_directories),
            'notebook': self.notebook_number,
            'with-augmentation': self.with_augmentation,
        }.items()

    @property
    def alpha(self) -> float:
        """
        Get the stepsize alpha.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """
        Set the stepsize alpha.
        """
        if not isinstance(alpha, float):
            raise TypeError('alpha needs to be a floating point number')
        if alpha <= 0:
            raise ValueError('the stepsize needs to be positive')
        self._alpha = alpha

    @property
    def base_data_directory(self) -> str:
        """
        Get the base_data_directory.
        """
        return self._base_data_directory

    @base_data_directory.setter
    def base_data_directory(self, path: str):
        """
        Set the base_data_directory.
        """
        self._base_data_directory = path

    @property
    def base_output_directory(self) -> str:
        """
        Get the base_output_directory.
        """
        return self._base_output_directory

    @base_output_directory.setter
    def base_output_directory(self, path: str):
        """
        Set the base_output_directory.
        """
        self._base_output_directory = path

    @property
    def notebook_number(self) -> int:
        """
        Get notebook number.
        """
        return self._notebook_number

    @notebook_number.setter
    def notebook_number(self, notebook: int):
        """
        Set the notebook number.
        """
        if notebook < 1:
            raise ValueError('notebook can not be zero '
                             f'or negative ({notebook})')
        self._notebook_number = notebook

    @property
    def model_version(self) -> str:
        """
        Get the model version name.
        """
        return self._model_version

    @model_version.setter
    def model_version(self, model_version: str):
        """
        Set the model version name.
        """
        self._model_version = model_version

    @property
    def model_file(self):
        """
        Get the location of the model file.
        """
        return os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_weights_{self.notebook_number}'
            f'_{self.model_version}'
            f'_tf{self.tf_compat}'
            f'_{self.image_dimensions[0]}'
            f'_{self.image_dimensions[1]}'
            '.keras')

    @property
    def history_file(self):
        """
        Get the history file.
        """
        return os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_history_{self.notebook_number}'
            f'_{self.model_version}'
            f'_tf{self.tf_compat}'
            f'_{self.image_dimensions[0]}'
            f'_{self.image_dimensions[1]}'
            '.pickle')

    @property
    def settings_file(self):
        """
        Get the settings file.
        """
        return os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_settings_{self.notebook_number}_{self.model_version}'
            f'_{self.image_dimensions[0]}_{self.image_dimensions[1]}.yaml')

    @property
    def color_mode(self) -> str:
        """
        Get the color_mode.
        """
        return self._color_mode

    @color_mode.setter
    def color_mode(self, mode: str):
        """
        Set the color_mode.
        """
        if not isinstance(mode, str):
            raise TypeError('mode must be a string')
        if mode not in ['rgb', 'grayscale']:
            raise ValueError("color_mode must be either 'rgb' or 'grayscale'")
        self._color_mode = mode

    @property
    def image_dimensions(self) -> Tuple[int, int]:
        """
        Get the image dimensions.
        """
        return self._image_dimensions

    @image_dimensions.setter
    def image_dimensions(self, dimensions: Tuple[int, int]):
        """
        Set the image dimensions.
        """
        if not (isinstance(dimensions, tuple) and
                len(dimensions) == 2 and
                all(isinstance(dim, int) for dim in dimensions)):
            raise TypeError('image dimensions needs to be a tuple')
        if not all(x > 0 for x in dimensions):
            raise ValueError('image dimensions need to be positive')
        self._image_dimensions = copy.deepcopy(dimensions)

    @property
    def load_history_from_file(self) -> bool:
        """
        Load history from file.
        """
        return self._load_history_from_file

    @load_history_from_file.setter
    def load_history_from_file(self, load_history: bool):
        """
        Load history from file.
        """
        self._load_history_from_file = load_history

    @property
    def load_model_from_file(self) -> bool:
        """
        Load model from file.
        """
        return self._load_model_from_file

    @load_model_from_file.setter
    def load_model_from_file(self, load_model: bool):
        """
        Load model from file.
        """
        self._load_model_from_file = load_model

    @property
    def epochs(self) -> int:
        """
        The number of epochs.
        """
        return self._epochs

    @epochs.setter
    def epochs(self, epochs: int):
        """
        Set the number of epochs.
        """
        if not isinstance(epochs, int):
            raise TypeError('epochs must be int')
        if epochs < 1:
            raise ValueError('epochs needs to be a positive integer')
        self._epochs = epochs

    @property
    def lion_directories(self) -> list[str]:
        """
        The directories containing lion images.
        """
        return [os.path.join(self.base_data_directory, lion)
                for lion in self._lion_directories]

    @lion_directories.setter
    def lion_directories(self, lions: list[str]):
        """
        Set the lion directories.
        """
        self._lion_directories = copy.deepcopy(lions)

    @property
    def no_lion_directories(self) -> list[str]:
        """
        The directories containing no_lion images.
        """
        return [os.path.join(self.base_data_directory, no_lion)
                for no_lion in self._no_lion_directories]

    @no_lion_directories.setter
    def no_lion_directories(self, no_lions: list[str]):
        """
        Set the no_lion directories.
        """
        self._no_lion_directories = copy.deepcopy(no_lions)

    @property
    def model_function_name(self) -> str:
        """
        Get the model function name.
        """
        return self._model_function_name

    @model_function_name.setter
    def model_function_name(self, name: str):
        """
        Set the model function name.
        """
        if name not in __MODEL_FUNCTIONS__:
            raise ValueError(f'unknown model function name {name}')
        self._model_function_name = name

    @property
    def with_augmentation(self) -> bool:
        """
        Get whether to augment training data.
        """
        return self._with_augmentation

    @with_augmentation.setter
    def with_augmentation(self, with_augmentation: bool):
        """
        Set whether to use augment training data.
        """
        self._with_augmentation = with_augmentation

    @property
    def batch_size(self) -> int:
        """
        Get the batch size.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """
        Set the batch size.
        """
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be int')
        if batch_size <= 0:
            raise ValueError('the batch-size needs to be a positive number')
        self._batch_size = batch_size

    @property
    def tf_compat(self) -> str:
        """
        Get the tensorflow compatibility version.

        Tensorflow changed their keras model file format from 2.15 to 2.17.
        Model files produced with tensorflow >= 2.15 to < 2.17 cannot be read
        with tensorflow >= 2.17. Model files therefore will be either in '2.15'
        or in '2.17' format.
        """
        return self._tf_compat

    @tf_compat.setter
    def tf_compat(self, compat: str):
        """
        Set the tensorflow compatibility version.

        This is either '2.15' or '2.17'.
        """
        if not isinstance(compat, str):
            raise TypeError('tf compat needs to be a string')
        if compat not in ['2.15', '2.17']:
            raise ValueError('tf compat needs to be in [2.15, 2.17]')
        self._tf_compat = compat
