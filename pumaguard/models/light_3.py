"""
The light-3 model.
"""

from typing import (
    Tuple,
)

import keras  # type: ignore

from pumaguard.model import (
    Model,
)


class LightModel03(Model):
    """
    Ultra-light CNN model for binary image classification (<10k params).
    """

    color_mode = 'rhb'

    def light_model_3(self, image_dimensions: Tuple[int, int]) -> keras.Model:
        """
        Ultra-light CNN model for binary image classification (<10k params).
        """
        return keras.Sequential([
            keras.Input(shape=(*image_dimensions, 3)),
            keras.layers.Conv2D(
                8,
                (3, 3),
                activation='relu',
                padding='same',
                strides=(2, 2)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(
                16,
                (3, 3),
                activation='relu',
                padding='same',
                strides=(2, 2)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
