"""
This script classifies images.
"""

# pylint: disable=redefined-outer-name

import argparse
import datetime
import os
import shutil
import sys
import tempfile

import numpy as np
import keras  # type: ignore

from pumaguard.presets import Presets
from pumaguard.traininghistory import TrainingHistory
from pumaguard.utils import (
    create_model,
    initialize_tensorflow,
)


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
    Classify the images and print out the result.
    """
    if presets.model_version == 'pre-trained':
        color_model = 'rgb'
    else:
        color_model = 'grayscale'
    print(f'Using color_mode \'{color_model}\'')

    print(f'Loading images from {image_path}')
    start_time = datetime.datetime.now()
    verification_dataset = keras.preprocessing.image_dataset_from_directory(
        image_path,
        batch_size=presets.batch_size,
        image_size=presets.image_dimensions,
        shuffle=True,
        color_mode=color_model,
    )
    end_time = datetime.datetime.now()
    print(f'Loaded images, {get_duration(start_time, end_time)} seconds')

    all_predictions = []
    all_labels = []

    for images, labels in verification_dataset:
        print('Starting classification of batch '
              f'of {presets.batch_size} images')
        start_time = datetime.datetime.now()
        predictions = model.predict(images)
        end_time = datetime.datetime.now()
        print('Classification took '
              f'{get_duration(start_time, end_time)} seconds')
        all_predictions.extend(predictions)
        all_labels.extend(labels)

    tmp = [a.tolist() for a in all_predictions]
    tmp2 = []
    for a in tmp:
        tmp2.extend(a)
    all_predictions = tmp2

    tmp = [a.numpy().tolist() for a in all_labels]
    all_labels = tmp

    # Calculate the percentage of correctly classified images
    correct_predictions = np.sum(np.round(all_predictions) == all_labels)
    total_images = len(all_labels)
    accuracy = correct_predictions / total_images * 100

    print(f"Percentage of correctly classified images: {accuracy:.2f}%")

    for images, labels in iter(verification_dataset):
        # Predict the labels for the images
        predictions = model.predict(images)
        for i in range(len(images)):
            print(
                f'Predicted: {100*(1 - predictions[i][0]):6.2f}% lion '
                f'({"lion" if predictions[i][0] < 0.5 else "no lion"}), '
                f'Actual: {"lion" if labels[i] == 0 else "no lion"}')


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
        'image',
        metavar='FILE[:{lion,no-lion}]',
        help=('An image to classify with an optional label. '
              'If the label is missing then "lion" is assumed.'),
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
    opts="-h --help --notebook"

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi

    case "${prev}" in
        --notebook)
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


def split_argument(image: str) -> tuple[str, str]:
    """
    Split a FILE:[{lion,no-lion}] command line argument into filename and
    label.

    Arguments:
        image -- The FILE:[{lion.no-lion}] command line argument.

    Raises:
        ValueError: If an unknown syntax was used for the argument.

    Returns:
        A (filename, label) tuple.
    """
    parsed = image.split(':')
    if len(parsed) == 1:
        filename = parsed[0]
        label = 'lion'
    elif len(parsed) == 2:
        filename, label = parsed
        if label not in ['lion', 'no-lion']:
            raise ValueError(
                f'unknown category {label}'
            )
    else:
        raise ValueError(
            'please use FILE[:{lion,no-lion}]'
        )
    return (filename, label)


def main():
    """
    Main entry point
    """

    options = parse_commandline()

    with tempfile.TemporaryDirectory() as workdir:
        images = {}
        for image in options.image:
            filename, label = split_argument(image)
            if label not in images:
                images[label] = []
            images[label].append(filename)

        for label in images:  # pylint: disable=consider-using-dict-items
            os.makedirs(os.path.join(workdir, label))
            for filename in images[label]:
                shutil.copy(filename, os.path.join(workdir, label))

        presets = Presets(options.notebook)
        full_history = TrainingHistory(presets)

        best_accuracy, best_val_accuracy, \
            best_loss, best_val_loss, best_epoch = \
            full_history.get_best_epoch('accuracy')
        print(f'Total time {sum(full_history.history["duration"])} '
              f'for {len(full_history.history["accuracy"])} epochs')
        print(f'Best epoch {best_epoch} - accuracy: {best_accuracy:.4f} - '
              f'val_accuracy: {best_val_accuracy:.4f} - loss: {best_loss:.4f} '
              f'- val_loss: {best_val_loss:.4f}')

        distribution_strategy = initialize_tensorflow()
        model = create_model(presets, distribution_strategy)
        classify_images(presets, model, workdir)
