"""
The pre-trained model.
"""

from typing import (
    Tuple,
)

import keras  # type: ignore


def pre_trained_model(image_dimensions: Tuple[int, int]) -> keras.src.Model:
    """
    The pre-trained model (Xception).

    Returns:
        The model.
    """
    base_model = keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(*image_dimensions, 3),
    )

    # We do not want to change the weights in the Xception model (imagenet
    # weights are frozen)
    base_model.trainable = False

    # Average pooling takes the 2,048 outputs of the Xeption model and brings
    # it into one output. The sigmoid layer makes sure that one output is
    # between 0-1. We will train all parameters in these last two layers
    return keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
