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
            self.lion_directories = [
                f'{self.base_data_directory}/lion',
                f'{self.base_data_directory}/cougar',
            ]
            self.no_lion_directories = [
                f'{self.base_data_directory}/no_lion',
                f'{self.base_data_directory}/nocougar',
            ]
        elif self.notebook_number == 7:
            self.image_dimensions = (512, 512)  # height, width
            self.with_augmentation = False
            self.batch_size = 16
            self.model_version = "light-2"
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

        self.model_file = os.path.realpath(
            f'{self.base_output_directory}/'
            f'model_weights_{self.notebook_number}_{self.model_version}'
            f'_{self.image_dimensions[0]}_{self.image_dimensions[1]}.keras')
        self.history_file = os.path.realpath(
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
    def load_history_from_file(self) -> bool:
        """
        Load history from file.
        """
        return self.__load_history_from_file

    @property
    def load_model_from_file(self) -> bool:
        """
        Load model from file.
        """
        return self.__load_model_from_file

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
