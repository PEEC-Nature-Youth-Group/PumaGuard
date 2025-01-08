"""
This script trains a model.
"""

import argparse
import datetime
import logging
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
        '--notebook',
        help='The notebook number',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--epochs',
        help='How many epochs to train',
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
            print_bash_completion()
            sys.exit(0)
        else:
            raise ValueError(f'unknown completion {options.completion}')
    return options


def print_bash_completion():
    """
    Print bash completion script.
    """
    print('''#!/bin/bash

_pumaguard_train_completions() {
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

complete -F _pumaguard_train_completions pumaguard.pumaguard-train
complete -F _pumaguard_train_completions pumaguard-train
''')


def main():
    """
    The main entry point.
    """
    logging.basicConfig(level=logging.INFO)

    options = parse_commandline()
    presets = Presets(options.notebook)

    if options.epochs:
        presets.epochs = options.epochs

    if options.no_load_previous_session:
        presets.load_history_from_file = False
        presets.load_model_from_file = False
    else:
        presets.load_history_from_file = True
        presets.load_model_from_file = True

    print(f'Model file   {presets.model_file}')
    print(f'History file {presets.history_file}')

    work_directory = tempfile.mkdtemp(prefix='pumaguard-work-')
    organize_data(presets=presets, work_directory=work_directory)

    if presets.model_version == 'pre-trained':
        presets.color_mode = 'rgb'
    else:
        presets.color_mode = 'grayscale'
    print(f'Using color_mode \'{presets.color_mode}\'')

    print(f'image dimensions {presets.image_dimensions}')

    training_dataset, validation_dataset = create_datasets(
        presets, work_directory, presets.color_mode)

    full_history = TrainingHistory(presets)

    best_accuracy, best_val_accuracy, best_loss, best_val_loss, \
        best_epoch = full_history.get_best_epoch('accuracy')
    print(f'Total time {sum(full_history.history["duration"])} '
          f'for {len(full_history.history["accuracy"])} epochs')
    print(f'Best epoch {best_epoch} - accuracy: {best_accuracy:.4f}'
          f'- val_accuracy: {best_val_accuracy:.4f} - loss: '
          f'{best_loss:.4f} - val_loss: {best_val_loss:.4f}')

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
