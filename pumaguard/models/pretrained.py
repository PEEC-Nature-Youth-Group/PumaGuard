"""
The pre-trained model.
"""

from typing import (
    Tuple,
)

import keras  # type: ignore

from pumaguard.model import (
    Model,
)


class PretrainedModel(Model):
    """
    The pre-trained model (Xception).
    """

    @property
    def model_name(self) -> str:
        """
        Get the model name.
        """
        return 'pretrained'

    @property
    def color_mode(self) -> str:
        """
        Get the color mode.
        """
        return 'rgb'

    @property
    def model_type(self) -> str:
        """
        Get the model type.
        """
        return 'pre-trained'

    def raw_model(self, image_dimensions: Tuple[int, int]) -> keras.Model:
        """
        The pre-trained model (Xception).
        """
        base_model = keras.applications.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(*image_dimensions, 3),
        )

        # We do not want to change the weights in the Xception model (imagenet
        # weights are frozen)
        base_model.trainable = False

        # Average pooling takes the 2,048 outputs of the Xception model and
        # brings it into one output. The sigmoid layer makes sure that one
        # output is between 0-1. We will train all parameters in these last
        # two layers
        return keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
