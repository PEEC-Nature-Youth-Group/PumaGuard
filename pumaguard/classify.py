"""
This script classifies images.
"""

# pylint: disable=redefined-outer-name

import argparse
import datetime
import numpy as np
import tensorflow as tf  # type: ignore
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
        "images",
        help=('The path to the folder containing the images. '
              'The folder needs to subfolders, "lion" and "no lion" '
              'that contain the images.'),
        type=str,
    )
    return parser.parse_args()


def main():
    """
    Main entry point
    """
    print("Tensorflow version " + tf.__version__)

    options = parse_commandline()
    presets = Presets(options.notebook)

    full_history = TrainingHistory(presets)

    best_accuracy, best_val_accuracy, best_loss, best_val_loss, best_epoch = \
        full_history.get_best_epoch('accuracy')
    print(f'Total time {sum(full_history.history["duration"])} '
          f'for {len(full_history.history["accuracy"])} epochs')
    print(f'Best epoch {best_epoch} - accuracy: {best_accuracy:.4f} - '
          f'val_accuracy: {best_val_accuracy:.4f} - loss: {best_loss:.4f} '
          f'- val_loss: {best_val_loss:.4f}')

    distribution_strategy = initialize_tensorflow()
    model = create_model(presets, distribution_strategy)
    classify_images(presets, model, options.images)
