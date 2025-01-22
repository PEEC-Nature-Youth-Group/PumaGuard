"""
This script trains a model.
"""

import argparse
import datetime
import logging
import os
import shutil
import sys
import tempfile

import keras  # type: ignore
import matplotlib.pyplot as plt
import yaml

from pumaguard.presets import (
    Presets,
)
from pumaguard.traininghistory import (
    TrainingHistory,
)
from pumaguard.utils import (
    create_datasets,
    create_model,
    initialize_tensorflow,
    organize_data,
)

logger = logging.getLogger('PumaGuard')


def plot_training_progress(filename, full_history):
    """
    Plot the training progress and store in file.
    """
    plt.figure(figsize=(18, 10))
    plt.subplot(1, 2, 1)
    plt.plot(full_history.history['accuracy'], label='Training Accuracy')
    plt.plot(full_history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(full_history.history['loss'], label='Training Loss')
    plt.plot(full_history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')

    print('Created plot of learning history')
    plt.savefig(filename)


def train_model(training_dataset,
                validation_dataset,
                full_history,
                presets: Presets,
                model: keras.src.Model):
    """
    Train the model.
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=presets.model_file,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.75,  # New lr = lr * factor.
        patience=50,
        verbose=1,
        mode='min',
        min_lr=1e-8,  # Lower bound on the learning rate.
    )

    print(f'Training for {presets.epochs} epochs')
    start_time = datetime.datetime.now()
    print(start_time)
    model.fit(
        training_dataset,
        epochs=presets.epochs,
        validation_data=validation_dataset,
        callbacks=[
            checkpoint,
            reduce_learning_rate,
            full_history,
        ]
    )
    end_time = datetime.datetime.now()
    print(end_time)

    duration = (end_time - start_time).total_seconds()
    print(f'This run took {duration} seconds')

    if 'duration' not in full_history.history:
        full_history.history['duration'] = []
    full_history.history['duration'].append(duration)

    print(f'total time {sum(full_history.history["duration"])} '
          f'for {len(full_history.history["accuracy"])} epochs')


def configure_subparser(parser: argparse.ArgumentParser):
    """
    Return Parser the command line.
    """
    parser.add_argument(
        '--lions',
        help='Directory with lion images',
        nargs='+',
    )
    parser.add_argument(
        '--no-lions',
        help='Directory with images not showing lions',
        nargs='+',
    )
    parser.add_argument(
        '--epochs',
        help='How many epochs to train.',
        type=int,
    )
    parser.add_argument(
        '--model-output',
        help='The output folder for the new model.',
        type=str,
    )
    parser.add_argument(
        '--no-load-previous-session',
        help='Do not load previous training session from file',
        action='store_true',
    )
    parser.add_argument(
        '--dump-settings',
        help='Print current settings to standard output',
        action='store_true',
    )


def print_training_stats(full_history: TrainingHistory):
    """
    Print some stats of training.
    """
    print(f'Total time {sum(full_history.history["duration"])} for '
          f'{len(full_history.history["accuracy"])} epochs')

    best_accuracy, best_val_accuracy, best_loss, best_val_loss, \
        best_epoch = full_history.get_best_epoch('accuracy')
    print(f'Best accuracy - epoch {best_epoch} - accuracy: '
          f'{best_accuracy:.4f} - val_accuracy: {best_val_accuracy:.4f} - '
          f'loss: {best_loss:.4f} - val_loss: {best_val_loss:.4f}')

    best_accuracy, best_val_accuracy, best_loss, best_val_loss, \
        best_epoch = full_history.get_best_epoch('val_accuracy')
    print(f'Best val_accuracy - epoch {best_epoch} - accuracy: '
          f'{best_accuracy:.4f} - val_accuracy: {best_val_accuracy:.4f} '
          f'- loss: {best_loss:.4f} - val_loss: {best_val_loss:.4f}')

    plot_training_progress('training-progress.png', full_history=full_history)


def main(options: argparse.Namespace):
    """
    The main entry point.
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
        else os.getenv('PUMAGUARD_DATA_DIRECTORY', default=None)
    if data_path is not None:
        logger.debug('setting data path to %s', data_path)
        presets.base_data_directory = data_path
        logger.warning('data path was specified, not using '
                       'lion/no_lion paths from presets')
        presets.lion_directories = []
        presets.no_lion_directories = []

    lion_directories = options.lions if options.lions else []
    if len(lion_directories) > 0:
        presets.lion_directories = [
            os.path.relpath(path, presets.base_data_directory)
            for path in lion_directories
        ]

    no_lion_directories = options.no_lions if options.no_lions else []
    if len(no_lion_directories) > 0:
        presets.no_lion_directories = [
            os.path.relpath(path, presets.base_data_directory)
            for path in no_lion_directories
        ]

    logger.debug('getting lion images from    %s', presets.lion_directories)
    logger.debug('getting no-lion images from %s', presets.no_lion_directories)

    if options.epochs:
        presets.epochs = options.epochs

    if options.no_load_previous_session:
        presets.load_history_from_file = False
        presets.load_model_from_file = False
    else:
        presets.load_history_from_file = True
        presets.load_model_from_file = True

    logger.info('model file    %s', presets.model_file)
    logger.info('history file  %s', presets.history_file)
    logger.info('settings file %s', presets.settings_file)

    if options.model_output:
        logger.debug('setting model output to %s', options.model_output)
        try:
            shutil.copy(presets.model_file, options.model_output)
            shutil.copy(presets.history_file, options.model_output)
        except FileNotFoundError:
            logger.warning('unable to find previous model; ignoring')
        presets.base_output_directory = options.model_output

    if options.dump_settings:
        print('# PumaGuard settings')
        print(yaml.safe_dump(
            dict(presets),
            # default_style='|',
            # canonical=False,
            indent=2,
        ))
        sys.exit(0)

    with open(presets.settings_file, 'w', encoding='utf-8') as fd:
        fd.write('# PumaGuard settings\n')
        fd.write(yaml.safe_dump(dict(presets)))

    work_directory = tempfile.mkdtemp(prefix='pumaguard-work-')
    organize_data(presets=presets, work_directory=work_directory)

    logger.info('using color_mode %s', presets.color_mode)
    logger.info('image dimensions %s', presets.image_dimensions)

    training_dataset, validation_dataset = create_datasets(
        presets, work_directory, presets.color_mode)

    full_history = TrainingHistory(presets)

    best_accuracy, best_val_accuracy, best_loss, best_val_loss, \
        best_epoch = full_history.get_best_epoch('accuracy')
    logger.info('Total time %.2f for %.2f epochs',
                sum(full_history.history["duration"]),
                len(full_history.history["accuracy"]))
    logger.info('Best epoch %d - accuracy: %.4f - val_accuracy: %.4f '
                '- loss: %.4f - val_loss: %.4f', best_epoch, best_accuracy,
                best_val_accuracy, best_loss, best_val_loss)

    distribution_strategy = initialize_tensorflow()
    model = create_model(presets, distribution_strategy)
    train_model(training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                model=model,
                presets=presets,
                full_history=full_history)

    print_training_stats(full_history)
