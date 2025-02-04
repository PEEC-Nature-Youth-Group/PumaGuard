"""
The Model factory.
"""

from typing import (
    Type,
)

from pumaguard.model import (
    Model,
)
from pumaguard.models import (
    __MODELS__,
)
from pumaguard.presets import (
    Preset,
)


def model_factory(presets: Preset) -> Type[Model]:
    """
    Get an instance of a model.
    """
    if presets.model_function_name not in __MODELS__:
        raise ValueError(f'unknown model {presets.model_function_name}')
    return __MODELS__[presets.model_function_name]
