"""
This script classifies images.
"""

# pylint: disable=redefined-outer-name

import argparse
import logging

from pumaguard.model import (
    model_factory,
)
from pumaguard.presets import (
    Preset,
)
from pumaguard.utils import (
    classify_image,
)

logger = logging.getLogger('PumaGuard')


def configure_subparser(parser: argparse.ArgumentParser):
    """
    Parse the commandline
    """
    parser.add_argument(
        'image',
        metavar='FILE',
        help='An image to classify.',
        nargs='*',
        type=str,
    )


def main(options: argparse.Namespace, presets: Preset):
    """
    Main entry point
    """

    logger.debug('loading model from %s', presets.model_file)
    model = model_factory(presets).model

    for image in options.image:
        prediction = classify_image(presets, model, image)
        if prediction >= 0:
            print(
                f'Predicted {image}: {100*(1 - prediction):6.2f}% lion '
                f'({"lion" if prediction < 0.5 else "no lion"})')
        else:
            logger.warning('predicted label < 0!')
