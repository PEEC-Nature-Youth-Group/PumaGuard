"""
The light-4 model.
"""

from typing import (
    Tuple,
)

import keras  # type: ignore


def light_model_4(image_dimensions: Tuple[int, int]) -> keras.Model:
    """
    Ultra light model.
    """
    return keras.Sequential([
        keras.Input(shape=(*image_dimensions, 3)),
        keras.layers.Conv2D(
            1, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
