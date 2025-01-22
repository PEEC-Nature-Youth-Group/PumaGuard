"""
This script classifies images.
"""

# pylint: disable=redefined-outer-name

import argparse
import logging
import os

import keras  # type: ignore

from pumaguard.presets import (
    Presets,
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
        '--data-path',
        help='Where the image data are stored',
    )
    parser.add_argument(
        'image',
        metavar='FILE',
        help='An image to classify.',
        nargs='*',
        type=str,
    )


def main(options: argparse.Namespace):
    """
    Main entry point
    """

    if options.debug:
        logger.setLevel(logging.DEBUG)

    presets = Presets(options.notebook)

    model_path = options.model_path if options.model_path \
        else os.getenv('PUMAGUARD_MODEL_PATH', default=None)
    if model_path is not None:
        logger.debug('setting model path to %s', model_path)
        presets.base_output_directory = model_path

    data_path = options.data_path if options.data_path \
        else os.getenv('PUMAGUARD_DATA_PATH', default=None)
    if data_path is not None:
        logger.debug('setting data path to %s', data_path)
        presets.base_data_directory = data_path

    try:
        os.stat(presets.model_file)
    except FileNotFoundError:
        logger.error('could not open model file %s', presets.model_file)
        raise

    logger.debug('loading model from %s', presets.model_file)
    model = keras.models.load_model(presets.model_file)

    for image in options.image:
        prediction = classify_image(presets, model, image)
        if prediction >= 0:
            print(
                f'Predicted {image}: {100*(1 - prediction):6.2f}% lion '
                f'({"lion" if prediction < 0.5 else "no lion"})')
        else:
            logger.warning('predicted label < 0!')
