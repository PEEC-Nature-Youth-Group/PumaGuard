"""
The light-2 model.
"""

import keras  # type: ignore

from pumaguard.presets import Presets


def light_model_2(presets: Presets) -> keras.src.Model:
    """
    Another attempt at a light model.
    """
    return keras.Sequential([
        keras.layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(*presets.image_dimensions, 1),
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
