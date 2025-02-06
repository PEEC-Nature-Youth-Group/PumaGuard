"""
The light-2 model.
"""

from typing import (
    Tuple,
)

import keras  # type: ignore

from pumaguard.model import (
    Model,
)


class LightModel2(Model):
    """
    Another light model.
    """

    @property
    def model_name(self) -> str:
        """
        Get the model name.
        """
        return 'light-2-model'

    @property
    def number_color_channels(self) -> int:
        """
        The number of color channels.
        """
        return 1

    @property
    def model_type(self) -> str:
        """
        Get the model type.
        """
        return 'light-2'

    def raw_model(self, image_dimensions: Tuple[int, int]) -> keras.Model:
        """
        Another attempt at a light model.
        """
        return keras.Sequential([
            keras.layers.Conv2D(
                32,
                (3, 3),
                activation='relu',
                input_shape=(*image_dimensions, self.number_color_channels),
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
