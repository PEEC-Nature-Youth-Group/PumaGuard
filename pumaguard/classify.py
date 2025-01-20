"""
This script classifies images.
"""

# pylint: disable=redefined-outer-name

import argparse
import logging
import os
import sys

import keras  # type: ignore

from pumaguard import (
    __VERSION__,
)
from pumaguard.presets import (
    Presets,
)
from pumaguard.utils import (
    classify_image,
    print_bash_completion,
)

logger = logging.getLogger('PumaGuard')


def parse_commandline() -> argparse.Namespace:
    """
    Parse the commandline

    Returns:
        argparse.NameSpace: The parsed options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        help='Debug the application',
        action='store_true',
    )
    parser.add_argument(
        '--notebook',
        help='The notebook number',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--model-path',
        help='Where the models are stored',
    )
    parser.add_argument(
        'image',
        metavar='FILE',
        help='An image to classify.',
        nargs='*',
        type=str,
    )
    parser.add_argument(
        '--completion',
        choices=['bash'],
        help='Print out bash completion script.',
    )
    options = parser.parse_args()
    if options.completion:
        if options.completion == 'bash':
            print_bash_completion('pumaguard-classify-completions.sh')
            sys.exit(0)
        else:
            raise ValueError(f'unknown completion {options.completion}')
    if not options.image:
        raise ValueError('missing FILE argument')
    return options


def main():
    """
    Main entry point
    """

    logging.basicConfig(level=logging.INFO)
    logger.info('PumaGuard Classify version %s', __VERSION__)
    options = parse_commandline()
    if options.debug:
        logger.setLevel(logging.DEBUG)

    presets = Presets(options.notebook)
    model_path = options.model_path if options.model_path \
        else os.getenv('PUMAGUARD_MODEL_PATH', default=None)
    if model_path is not None:
        logger.debug('setting model path to %s', model_path)
        presets.base_output_directory = model_path

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
            logger.warning('classification failed: file not found')
