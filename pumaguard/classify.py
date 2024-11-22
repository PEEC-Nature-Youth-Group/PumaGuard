"""
This script classifies images.
"""

# pylint: disable=redefined-outer-name

import argparse
import datetime
import os
import numpy as np
import tensorflow as tf  # type: ignore
import keras  # type: ignore

from pumaguard.presets import Presets
from pumaguard.traininghistory import TrainingHistory
from pumaguard.utils import (
    initialize_tensorflow,
)
from pumaguard.models.pretrained import pre_trained_model
from pumaguard.models.light import light_model
from pumaguard.models.light_2 import light_model_2


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


def create_model(presets: Presets,
                 distribution_strategy: tf.distribute.Strategy):
    """
    Create the model.
    """
    with distribution_strategy.scope():
        model_file_exists = os.path.isfile(presets.model_file)
        if presets.load_model_from_file and model_file_exists:
            os.stat(presets.model_file)
            print(f'Loading model from file {presets.model_file}')
            model = keras.models.load_model(presets.model_file)
            print('Loaded model from file')
        else:
            print('Creating new model')
            if presets.model_version == "pre-trained":
                print('Creating new Xception model')
                model = pre_trained_model(presets)
                print('Building pre-trained model')
                model.build(input_shape=(None, *presets.image_dimensions, 3))
                print('Compiling pre-trained model')
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                )
            elif presets.model_version == "light":
                print('Creating new light model')
                model = light_model(presets)
                print('Building light model')
                model.build(input_shape=(None, *presets.image_dimensions, 1))
                print('Compiling light model')
                model.compile(
                    optimizer=keras.optimizers.Adam(
                        learning_rate=presets.alpha),
                    loss=keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
                )
            elif presets.model_version == 'light-2':
                print('Creating new light-2 model')
                model = light_model_2(presets)
                print('Building light-2 model')
                model.build(input_shape=(None, *presets.image_dimensions, 1))
                print('Compiling light model')
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                )
            else:
                raise ValueError(
                    f'unknown model version {presets.model_version}')

            print(f'Number of layers in the model: {len(model.layers)}')
            model.summary()

    return model


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
