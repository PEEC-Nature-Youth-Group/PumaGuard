"""
The presets for each model.
"""

import copy
import os
from typing import (
    Callable,
    Tuple,
)

import yaml

from pumaguard.models import (
    light,
    light_2,
    light_3,
    pretrained,
)


class BasePreset():
    """
    Base class for Presets
    """

    __alpha = 1e-4
    __base_data_directory: str = 'undefined'
    __base_output_directory: str = 'undefined'
    __batch_size = 16
    __color_mode: str = 'undefined'
    __epochs = 300
    __image_dimension: Tuple[int, int] = (128, 128)
    __lion_directories: list[str] = []
    __load_history_from_file = True
    __load_model_from_file = True
    __model_function: Callable
    __model_version = 'undefined'
    __no_lion_directories: list[str] = []
    __notebook_number = -1
    __with_augmentation = False

    def __init__(self):
        self.base_data_directory = os.path.join(
            os.path.dirname(__file__), '../data')
        self.base_output_directory = os.path.join(
            os.path.dirname(__file__), '../models')

    def load(self, filename: str):
        """
        Load settings from YAML file.
        """
        with open(filename, encoding='utf-8') as fd:
            settings = yaml.safe_load(fd)
        self.notebook_number = settings.get('notebook-number', 1)
        self.epochs = settings.get('epochs', 1)
        dimensions = settings.get('image-dimensions', [0, 0])
        if not isinstance(dimensions, list) or \
            len(dimensions) != 2 or \
                not all(isinstance(d, int) for d in dimensions):
            raise ValueError(
                'expected image-dimensions to be a list of two integers')
        self.image_dimensions = tuple(dimensions)
        self.model_version = settings.get('model-version', 'undefined')
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

    @property
    def notebook_number(self) -> int:
        """
        Get notebook number.
        """
        return self.__notebook_number

    @notebook_number.setter
    def notebook_number(self, notebook: int):
        """
        Set the notebook number.
        """
        if notebook < 1:
            raise ValueError('notebook can not be zero '
                             f'or negative ({notebook})')
        self.__notebook_number = notebook

    @property
    def model_version(self) -> str:
        """
        Get the model version name.
        """
        return self.__model_version

    @model_version.setter
    def model_version(self, model_version: str):
        """
        Set the model version name.
        """
        self.__model_version = model_version

    @property
    def model_file(self):
        """
        Get the location of the model file.
        """
        return os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_weights_{self.notebook_number}_{self.model_version}'
            f'_{self.image_dimensions[0]}_{self.image_dimensions[1]}.keras')

    @property
    def history_file(self):
        """
        Get the history file.
        """
        return os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_history_{self.notebook_number}_{self.model_version}'
            f'_{self.image_dimensions[0]}_{self.image_dimensions[1]}.pickle')

    @property
    def color_mode(self) -> str:
        """
        Get the color_mode.
        """
        return self.__color_mode

    @color_mode.setter
    def color_mode(self, mode: str):
        """
        Set the color_mode.
        """
        if mode not in ['rgb', 'grayscale']:
            raise ValueError("color_mode must be either 'rgb' or 'grayscale'")
        self.__color_mode = mode

    @property
    def image_dimensions(self) -> Tuple[int, int]:
        """
        Get the image dimensions.
        """
        return self.__image_dimension

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

    @property
    def base_data_directory(self) -> str:
        """
        Get the base_data_directory.
        """
        return self.__base_data_directory

    @base_data_directory.setter
    def base_data_directory(self, path: str):
        """
        Set the base_data_directory.
        """
        self.__base_data_directory = path

    @property
    def base_output_directory(self) -> str:
        """
        Get the base_output_directory.
        """
        return self.__base_output_directory

    @base_output_directory.setter
    def base_output_directory(self, path: str):
        """
        Set the base_output_directory.
        """
        self.__base_output_directory = path

    @property
    def load_history_from_file(self) -> bool:
        """
        Load history from file.
        """
        return self.__load_history_from_file

    @load_history_from_file.setter
    def load_history_from_file(self, load_history: bool):
        """
        Load history from file.
        """
        self.__load_history_from_file = load_history

    @property
    def load_model_from_file(self) -> bool:
        """
        Load model from file.
        """
        return self.__load_model_from_file

    @load_model_from_file.setter
    def load_model_from_file(self, load_model: bool):
        """
        Load model from file.
        """
        self.__load_model_from_file = load_model

    @property
    def epochs(self) -> int:
        """
        The number of epochs.
        """
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs: int):
        """
        Set the number of epochs.
        """
        if epochs < 1:
            raise ValueError('epochs needs to be a positive integer')
        self.__epochs = epochs

    @property
    def lion_directories(self) -> list[str]:
        """
        The directories containing lion images.
        """
        return [os.path.join(self.base_data_directory, lion)
                for lion in self.__lion_directories]

    @lion_directories.setter
    def lion_directories(self, lions: list[str]):
        """
        Set the lion directories.
        """
        self.__lion_directories = copy.deepcopy(lions)

    @property
    def no_lion_directories(self) -> list[str]:
        """
        The directories containing no_lion images.
        """
        return [os.path.join(self.base_data_directory, no_lion)
                for no_lion in self.__no_lion_directories]

    @no_lion_directories.setter
    def no_lion_directories(self, no_lions: list[str]):
        """
        Set the no_lion directories.
        """
        self.__no_lion_directories = copy.deepcopy(no_lions)

    @property
    def model_function(self) -> Callable:
        """
        Get the model function.
        """
        return self.__model_function

    @model_function.setter
    def model_function(self, func: Callable):
        """
        Set the model function.
        """
        self.__model_function = func

    @property
    def with_augmentation(self) -> bool:
        """
        Get whether to augment training data.
        """
        return self.__with_augmentation

    @with_augmentation.setter
    def with_augmentation(self, with_augmentation: bool):
        """
        Set whether to use augment training data.
        """
        self.__with_augmentation = with_augmentation

    @property
    def batch_size(self) -> int:
        """
        Get the batch size.
        """
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """
        Set the batch size.
        """
        if batch_size <= 0:
            raise ValueError('the batch-size needs to be a positive number')
        self.__batch_size = batch_size

    @property
    def alpha(self) -> float:
        """
        Get the stepsize alpha.
        """
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """
        Set the stepsize alpha.
        """
        if alpha <= 0:
            raise ValueError('the stepsize needs to be positive')
        self.__alpha = alpha


class Preset01(BasePreset):
    """
    Preset 01
    """

    def __init__(self):
        super().__init__()
        self.notebook_number = 1

        self.epochs = 2_400
        self.image_dimensions = (128, 128)  # height, width
        self.with_augmentation = False
        self.batch_size = 16
        self.model_version = "light"
        self.model_function = light.light_model
        self.color_mode = 'grayscale'
        self.alpha = 1e-5
        self.lion_directories = [
            # 'lion_1',
            'lion',
        ]
        self.no_lion_directories = [
            # 'no_lion_1',
            'no_lion',
        ]


class Presets():
    """
    Presets for training.
    """

    __notebook_number = -1
    __color_mode: str = 'undefined'
    __load_model_from_file = True
    __load_history_from_file = True
    __epochs = 300
    __model_function: Callable
    __base_data_directory: str = 'undefined'
    __base_output_directory: str = 'undefined'
    __lion_directories: list[str] = []
    __no_lion_directories: list[str] = []

    def __init__(self, notebook_number: int = 1):
        self.base_data_directory = os.path.join(
            os.path.dirname(__file__), '../data')
        self.base_output_directory = os.path.join(
            os.path.dirname(__file__), '../models')

        self.notebook_number = notebook_number

        # Default step size.
        self.alpha = 1e-5

        # No changes below this line.
        if self.notebook_number == 1:
            self.epochs = 2_400
            self.image_dimensions = (128, 128)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "light"
            self.model_function = light.light_model
            self.color_mode = 'grayscale'
            self.alpha = 1e-5
            self.lion_directories = [
                # 'lion_1',
                'lion',
            ]
            self.no_lion_directories = [
                # 'no_lion_1',
                'no_lion',
            ]
        elif self.notebook_number == 2:
            self.epochs = 1_200
            self.image_dimensions = (256, 256)  # height, width
            self.with_augmentation = False
            self.batch_size = 32
            self.model_version = "light"
            self.model_function = light.light_model
            self.color_mode = 'grayscale'
            self.lion_directories = [
                # f'{base_data_directory}/lion_1',
                'lion',
            ]
            self.no_lion_directories = [
                # f'{base_data_directory}/no_lion_1',
                'no_lion',
            ]
        elif self.notebook_number == 3:
            self.epochs = 900
            self.image_dimensions = (256, 256)  # height, width
            self.with_augmentation = True
            self.batch_size = 32
            self.model_version = "light"
            self.model_function = light.light_model
            self.color_mode = 'grayscale'
            self.lion_directories = [
                'lion',
            ]
            self.no_lion_directories = [
                'no_lion',
            ]
        elif self.notebook_number == 4:
            self.image_dimensions = (128, 128)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.model_function = pretrained.pre_trained_model
            self.color_mode = 'rgb'
            self.lion_directories = [
                'lion_1',
            ]
            self.no_lion_directories = [
                'no_lion_1',
            ]
        elif self.notebook_number == 5:
            self.image_dimensions = (128, 128)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.model_function = pretrained.pre_trained_model
            self.color_mode = 'rgb'
            self.lion_directories = [
                'lion',
            ]
            self.no_lion_directories = [
                'no_lion',
            ]
        elif self.notebook_number == 6:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.model_function = pretrained.pre_trained_model
            self.color_mode = 'rgb'
            self.lion_directories = [
                'lion',
                'cougar',
            ]
            self.no_lion_directories = [
                'no_lion',
                'nocougar',
            ]
        elif notebook_number == 7:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.model_function = pretrained.pre_trained_model
            self.color_mode = 'rgb'
            self.lion_directories = [
                'lion',
                'cougar',
                'stable/angle 1/Lion',
                'stable/angle 2/Lion',
                'stable/angle 3/Lion',
                'stable/angle 4/lion',
            ]
            self.no_lion_directories = [
                'no_lion',
                'nocougar',
                'stable/angle 1/No Lion',
                'stable/angle 2/No Lion',
                'stable/angle 3/No Lion',
                'stable/angle 4/no lion',
            ]
        elif self.notebook_number == 8:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "light-2"
            self.model_function = light_2.light_model_2
            self.color_mode = 'grayscale'
            self.lion_directories = [
                'lion',
                'cougar',
            ]
            self.no_lion_directories = [
                'no_lion',
                'nocougar',
            ]
        elif self.notebook_number == 9:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "light-3"
            self.model_function = light_3.light_model_3
            self.color_mode = 'rgb'
            self.lion_directories = [
                'lion',
                'cougar',
                'stable/angle 1/Lion',
                'stable/angle 2/Lion',
                'stable/angle 3/Lion',
                'stable/angle 4/lion',
            ]
            self.no_lion_directories = [
                'no_lion',
                'nocougar',
                'stable/angle 1/No Lion',
                'stable/angle 2/No Lion',
                'stable/angle 3/No Lion',
                'stable/angle 4/no lion',
            ]
        else:
            raise ValueError(f'unknown notebook {self.notebook_number}')

    @property
    def notebook_number(self) -> int:
        """
        Get notebook number.
        """
        return self.__notebook_number

    @notebook_number.setter
    def notebook_number(self, notebook: int):
        """
        Set the notebook number.
        """
        if notebook < 1:
            raise ValueError('notebook can not be zero '
                             f'or negative ({notebook})')
        self.__notebook_number = notebook

    @property
    def model_file(self):
        """
        Get the location of the model file.
        """
        return os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_weights_{self.notebook_number}_{self.model_version}'
            f'_{self.image_dimensions[0]}_{self.image_dimensions[1]}.keras')

    @property
    def history_file(self):
        """
        Get the history file.
        """
        return os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_history_{self.notebook_number}_{self.model_version}'
            f'_{self.image_dimensions[0]}_{self.image_dimensions[1]}.pickle')

    @property
    def color_mode(self) -> str:
        """
        Get the color_mode.
        """
        return self.__color_mode

    @color_mode.setter
    def color_mode(self, mode: str):
        """
        Set the color_mode.
        """
        if mode not in ['rgb', 'grayscale']:
            raise ValueError("color_mode must be either 'rgb' or 'grayscale'")
        self.__color_mode = mode

    @property
    def base_data_directory(self) -> str:
        """
        Get the base_data_directory.
        """
        return self.__base_data_directory

    @base_data_directory.setter
    def base_data_directory(self, path: str):
        """
        Set the base_data_directory.
        """
        self.__base_data_directory = path

    @property
    def base_output_directory(self) -> str:
        """
        Get the base_output_directory.
        """
        return self.__base_output_directory

    @base_output_directory.setter
    def base_output_directory(self, path: str):
        """
        Set the base_output_directory.
        """
        self.__base_output_directory = path

    @property
    def load_history_from_file(self) -> bool:
        """
        Load history from file.
        """
        return self.__load_history_from_file

    @load_history_from_file.setter
    def load_history_from_file(self, load_history: bool):
        """
        Load history from file.
        """
        self.__load_history_from_file = load_history

    @property
    def load_model_from_file(self) -> bool:
        """
        Load model from file.
        """
        return self.__load_model_from_file

    @load_model_from_file.setter
    def load_model_from_file(self, load_model: bool):
        """
        Load model from file.
        """
        self.__load_model_from_file = load_model

    @property
    def epochs(self) -> int:
        """
        The number of epochs.
        """
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs: int):
        """
        Set the number of epochs.
        """
        if epochs < 1:
            raise ValueError('epochs needs to be a positive integer')
        self.__epochs = epochs

    @property
    def lion_directories(self) -> list[str]:
        """
        The directories containing lion images.
        """
        return [os.path.join(self.base_data_directory, lion)
                for lion in self.__lion_directories]

    @lion_directories.setter
    def lion_directories(self, lions: list[str]):
        """
        Set the lion directories.
        """
        self.__lion_directories = copy.deepcopy(lions)

    @property
    def no_lion_directories(self) -> list[str]:
        """
        The directories containing no_lion images.
        """
        return [os.path.join(self.base_data_directory, no_lion)
                for no_lion in self.__no_lion_directories]

    @no_lion_directories.setter
    def no_lion_directories(self, no_lions: list[str]):
        """
        Set the no_lion directories.
        """
        self.__no_lion_directories = copy.deepcopy(no_lions)

    @property
    def model_function(self) -> Callable:
        """
        Get the model function.
        """
        return self.__model_function

    @model_function.setter
    def model_function(self, func: Callable):
        """
        Set the model function.
        """
        self.__model_function = func
