"""
This script verifies models against a standard set of images.
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
        'image',
        metavar='FILE',
        help='An image to classify.',
        nargs='*',
        type=str,
    )


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

    verify_model(presets, model)
