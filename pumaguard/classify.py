"""
This script classifies images.
"""

# pylint: disable=redefined-outer-name

import argparse
import datetime
import logging
import os
import sys

import keras  # type: ignore
import numpy as np
from PIL import (
    Image,
)

from pumaguard import (
    __VERSION__,
)
from pumaguard.presets import (
    Presets,
)

logger = logging.getLogger('PumaGuard-Server')


def get_duration(start_time: datetime.datetime,
                 end_time: datetime.datetime) -> float:
    """
    Get duration between start and end time in seconds.

    Args:
        start_time (datetime.timezone): The start time.
        end_time (datetime.timezone): The end time.

    Returns:
        float: The duration in seconds.
    """
    duration = end_time - start_time
    return duration / datetime.timedelta(microseconds=1) / 1e6


def classify_images(presets: Presets, model: keras.Model, image_path: str):
    """
    Classify the image and print out the result.
    """
    print(f'Using color_mode \'{presets.color_mode}\'')
    print(f'Classifying image {image_path}')
    start_time = datetime.datetime.now()
    if presets.color_mode == 'rgb':
        img = Image.open(image_path).convert('RGB')
    else:
        img = Image.open(image_path).convert('L')
    img = img.resize(presets.image_dimensions)

    img_array = np.array(img)
    # img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    if presets.color_mode == 'grayscale':
        img_array = np.expand_dims(img_array, axis=-1)

    prediction = model.predict(img_array)

    end_time = datetime.datetime.now()
    print('Classification took '
          f'{get_duration(start_time, end_time)} seconds')

    print(
        f'Predicted {image_path}: {100*(1 - prediction[0][0]):6.2f}% lion '
        f'({"lion" if prediction[0][0] < 0.5 else "no lion"})')


def parse_commandline() -> argparse.Namespace:
    """
    Parse the commandline

    Returns:
        argparse.NameSpace: The parsed options.
    """
    parser = argparse.ArgumentParser()
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
            print_bash_completion()
            sys.exit(0)
        else:
            raise ValueError(f'unknown completion {options.completion}')
    if not options.image:
        raise ValueError('missing FILE argument')
    return options


def print_bash_completion():
    """
    Print bash completion script.
    """
    print('''#!/bin/bash

_pumaguard_classify_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-h --help --notebook --completion"

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi

    case "${prev}" in
        --notebook)
            return 0
            ;;
       --completion)
            opts="bash"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
         *)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

complete -F _pumaguard_classify_completions pumaguard.pumaguard-classify
complete -F _pumaguard_classify_completions pumaguard-classify
''')


def main():
    """
    Main entry point
    """

    logging.basicConfig(level=logging.INFO)
    logger.info('PumaGuard Server version %s', __VERSION__)
    options = parse_commandline()
    presets = Presets(options.notebook)
    model_path = options.model_path if options.model_path \
        else os.getenv('PUMAGUARD_MODEL_PATH', default=None)
    if model_path is not None:
        logger.debug('setting model path to %s', model_path)
        presets.base_output_directory = model_path
    model = keras.models.load_model(presets.model_file)
    for image in options.image:
        classify_images(presets, model, image)
