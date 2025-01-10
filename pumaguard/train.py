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
    print_bash_completion,
)

logger = logging.getLogger('PumaGuard-Server')


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


def parse_commandline() -> argparse.Namespace:
    """
    Parse the command line.
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
        '--model-output',
        help='The output folder for the new model.',
        type=str,
    )
    parser.add_argument(
        '--model-path',
        help='Where to load models from.',
    )
    parser.add_argument(
        '--data-directory',
        help='The base directory of the trainin data',
        type=str,
    )
    parser.add_argument(
        '--epochs',
        help='How many epochs to train.',
        type=int,
    )
    parser.add_argument(
        '--no-load-previous-session',
        help='Do not load previous training session from file',
        action='store_true',
    )
    parser.add_argument(
        '--completion',
        choices=['bash'],
        help='Print out bash completion script.',
    )
    options = parser.parse_args()
    if options.completion:
        if options.completion == 'bash':
            print_bash_completion('pumaguard-train-completions.sh')
            sys.exit(0)
        else:
            raise ValueError(f'unknown completion {options.completion}')
    return options


def main():
    """
    The main entry point.
    """
    logging.basicConfig(level=logging.INFO)

    options = parse_commandline()
    if options.debug:
        logger.setLevel(logging.DEBUG)

    presets = Presets(options.notebook)
    model_path = options.model_path if options.model_path \
        else os.getenv('PUMAGUARD_MODEL_PATH', default=None)
    if model_path is not None:
        logger.debug('setting model path to %s', model_path)
        presets.base_output_directory = model_path

    data_directory = options.data_directory if options.data_directory \
        else os.getenv('PUMAGUARD_DATA_DIRECTORY', default=None)
    if data_directory is not None:
        logger.debug('setting data directory to %s', data_directory)
        presets.base_data_directory = data_directory

    if options.epochs:
        presets.epochs = options.epochs

    if options.no_load_previous_session:
        presets.load_history_from_file = False
        presets.load_model_from_file = False
    else:
        presets.load_history_from_file = True
        presets.load_model_from_file = True

    logger.info('Model file   %s', presets.model_file)
    logger.info('History file %s', presets.history_file)

    if options.model_output:
        logger.debug('setting model output to %s', options.model_output)
        try:
            shutil.copy(presets.model_file, options.model_output)
            shutil.copy(presets.history_file, options.model_output)
        except FileNotFoundError:
            logger.warning('unable to find previous model; ignoring')
        presets.base_output_directory = options.model_output

    work_directory = tempfile.mkdtemp(prefix='pumaguard-work-')
    organize_data(presets=presets, work_directory=work_directory)

    if presets.model_version == 'pre-trained':
        presets.color_mode = 'rgb'
    else:
        presets.color_mode = 'grayscale'
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

    # Print some stats of training so far

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
