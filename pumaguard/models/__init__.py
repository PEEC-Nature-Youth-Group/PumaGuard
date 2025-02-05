"""
The models.
"""

from pumaguard.models import (
    light,
    light_2,
    light_3,
    light_4,
    pretrained,
)

__MODEL_FUNCTIONS__ = {
    'pretrained': pretrained.pre_trained_model,
    'light-model': light.light_model,
    'light-2-model': light_2.light_model_2,
    'light-3-model': light_3.light_model_3,
    'light-4-model': light_4.light_model_4,
}
