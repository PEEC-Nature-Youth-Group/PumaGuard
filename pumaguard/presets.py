"""
The presets for each model.
"""

import os


class Presets():
    """
    Presets for training.
    """

    __color_mode: str = 'undefined'
    __load_model_from_file = True
    __load_history_from_file = True
    __epochs = 300
    __base_data_directory: str = 'undefined'
    __base_output_directory: str = 'undefined'

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
            self.__epochs = 2_400
            self.image_dimensions = (128, 128)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "light"
            self.__color_mode = 'grayscale'
            self.alpha = 1e-5
            self.lion_directories = [
                # f'{self.base_data_directory}/lion_1',
                f'{self.base_data_directory}/lion',
            ]
            self.no_lion_directories = [
                # f'{self.base_data_directory}/no_lion_1',
                f'{self.base_data_directory}/no_lion',
            ]
        elif self.notebook_number == 2:
            self.__epochs = 1_200
            self.image_dimensions = (256, 256)  # height, width
            self.with_augmentation = False
            self.batch_size = 32
            self.model_version = "light"
            self.__color_mode = 'grayscale'
            self.lion_directories = [
                # f'{base_data_directory}/lion_1',
                f'{self.base_data_directory}/lion',
            ]
            self.no_lion_directories = [
                # f'{base_data_directory}/no_lion_1',
                f'{self.base_data_directory}/no_lion',
            ]
        elif self.notebook_number == 3:
            self.__epochs = 900
            self.image_dimensions = (256, 256)  # height, width
            self.with_augmentation = True
            self.batch_size = 32
            self.model_version = "light"
            self.__color_mode = 'grayscale'
            self.lion_directories = [
                f'{self.base_data_directory}/lion',
            ]
            self.no_lion_directories = [
                f'{self.base_data_directory}/no_lion',
            ]
        elif self.notebook_number == 4:
            self.image_dimensions = (128, 128)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.__color_mode = 'rgb'
            self.lion_directories = [
                f'{self.base_data_directory}/lion_1',
            ]
            self.no_lion_directories = [
                f'{self.base_data_directory}/no_lion_1',
            ]
        elif self.notebook_number == 5:
            self.image_dimensions = (128, 128)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.__color_mode = 'rgb'
            self.lion_directories = [
                f'{self.base_data_directory}/lion',
            ]
            self.no_lion_directories = [
                f'{self.base_data_directory}/no_lion',
            ]
        elif self.notebook_number == 6:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.__color_mode = 'rgb'
            self.lion_directories = [
                f'{self.base_data_directory}/lion',
                f'{self.base_data_directory}/cougar',
            ]
            self.no_lion_directories = [
                f'{self.base_data_directory}/no_lion',
                f'{self.base_data_directory}/nocougar',
            ]
        elif notebook_number == 7:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "pre-trained"
            self.__color_mode = 'rgb'
            self.lion_directories = [
                f'{self.base_data_directory}/lion',
                f'{self.base_data_directory}/cougar',
                f'{self.base_data_directory}/stable/angle 1/Lion',
                f'{self.base_data_directory}/stable/angle 2/Lion',
                f'{self.base_data_directory}/stable/angle 3/Lion',
                f'{self.base_data_directory}/stable/angle 4/lion',
            ]
            self.no_lion_directories = [
                f'{self.base_data_directory}/no_lion',
                f'{self.base_data_directory}/nocougar',
                f'{self.base_data_directory}/stable/angle 1/No Lion',
                f'{self.base_data_directory}/stable/angle 2/No Lion',
                f'{self.base_data_directory}/stable/angle 3/No Lion',
                f'{self.base_data_directory}/stable/angle 4/no lion',
            ]
        elif self.notebook_number == 8:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "light-2"
            self.__color_mode = 'rgb'
            self.lion_directories = [
                f'{self.base_data_directory}/lion',
                f'{self.base_data_directory}/cougar',
            ]
            self.no_lion_directories = [
                f'{self.base_data_directory}/no_lion',
                f'{self.base_data_directory}/nocougar',
            ]
        else:
            raise ValueError(f'Unknown notebook {self.notebook_number}')

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
