"""
The light-3 model.
"""

import keras  # type: ignore

from pumaguard.presets import (
    Presets,
)


def light_model_3(presets: Presets) -> keras.src.Model:
    """
    Ultra-light CNN model for binary image classification (<10k params).
    """
    return keras.Sequential([
        keras.layers.Conv2D(
            8,
            (3, 3),
            activation='relu',
            input_shape=(*presets.image_dimensions, 3),
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
