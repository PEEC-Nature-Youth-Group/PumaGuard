"""
The pre-trained model.
"""

import keras  # type: ignore

from pumaguard.presets import Presets


def pre_trained_model(presets: Presets) -> keras.src.Model:
    """
    The pre-trained model (Xception).

    Returns:
        The model.
    """
    base_model = keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(*presets.image_dimensions, 3),
    )

    print(f'Number of layers in the base model: {len(base_model.layers)}')
    print(f'shape of output layer: {base_model.layers[-1].output_shape}')

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
