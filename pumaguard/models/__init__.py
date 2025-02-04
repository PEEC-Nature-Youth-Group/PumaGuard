"""
The models.
"""

from pumaguard.models import (
    light,
    light_2,
    light_3,
    pretrained,
)

__MODELS__ = {
    'pretrained': pretrained.PretrainedModel,
    'light-model': light.LightModel01,
    'light-2-model': light_2.LightModel02,
    'light-3-model': light_3.LightModel03,
}
