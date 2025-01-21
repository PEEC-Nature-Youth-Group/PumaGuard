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
        '--settings',
        help='Load presets from file',
        type=str,
    )
    parser.add_argument(
        '--model-path',
        help='Where the models are stored',
    )
    parser.add_argument(
        '--data-path',
        help='Where the image data are stored',
    )
    parser.add_argument(
        '--completion',
        choices=['bash'],
        help='Print out bash completion script.',
    )
    parser.add_argument(
        '--verify',
        help='Use verification data to calculate accuracy of model.',
        action='store_true',
    )
    parser.add_argument(
        'image',
        metavar='FILE',
        help='An image to classify.',
        nargs='*',
        type=str,
    )
    options = parser.parse_args()
    if options.completion:
        if options.completion == 'bash':
            print_bash_completion('pumaguard-classify-completions.sh')
            sys.exit(0)
        else:
            raise ValueError(f'unknown completion {options.completion}')
    if not options.image and not options.verify:
        raise ValueError('missing FILE argument')
    return options


def verify_model(presets: Presets, model: keras.Model):
    """
    Verify a model by calculating its accuracy across a standard set of images.
    """
    logger.info('verifying model')
    lion_directory = os.path.join(
        presets.base_data_directory, 'verification', 'lion')
    lions = os.listdir(lion_directory)
    no_lion_directory = os.path.join(
        presets.base_data_directory, 'verification', 'no lion')
    no_lions = os.listdir(no_lion_directory)
    confusion = {
        'TP': 0.0, 'TN': 0.0, 'FP': 0.0, 'FN': 0.0,
    }
    for lion in lions:
        logger.debug('classifying %s', os.path.join(lion_directory, lion))
        prediction = classify_image(presets, model, os.path.join(
            lion_directory, lion))
        if prediction >= 0:
            print(f'Predicted {lion}: {100*(1 - prediction):6.2f}% lion')
            confusion['TP'] += 1 - prediction
            confusion['FN'] += prediction
        else:
            logger.warning('predicted label < 0!')
    for no_lion in no_lions:
        logger.debug('classifying %s', os.path.join(
            no_lion_directory, no_lion))
        prediction = classify_image(presets, model, os.path.join(
            no_lion_directory, no_lion))
        if prediction >= 0:
            print(
                f'Predicted {no_lion}: {100*(1 - prediction):6.2f}% lion')
            confusion['TN'] += prediction
            confusion['FP'] += 1 - prediction
        else:
            logger.warning('predicted label < 0!')
    total = sum(confusion.values())
    logger.debug(confusion)
    logger.debug(total)
    logger.debug('%d lions and %d no lions', len(lions), len(no_lions))
    # if abs(total - len(lions) + len(no_lions)) > 0.1:
    #     logger.error('some images could not be classified')
    #     sys.exit(1)
    accuracy = (confusion['TP'] + confusion['TN']) / total
    print(f'accuracy = {100 * accuracy:.2f}%')


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

    if options.verify:
        verify_model(presets, model)
    else:
        for image in options.image:
            prediction = classify_image(presets, model, image)
            if prediction >= 0:
                print(
                    f'Predicted {image}: {100*(1 - prediction):6.2f}% lion '
                    f'({"lion" if prediction < 0.5 else "no lion"})')
            else:
                logger.warning('predicted label < 0!')
