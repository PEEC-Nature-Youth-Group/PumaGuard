"""
This script trains a model.
"""

import argparse
import datetime
import os
import tempfile

import keras  # type: ignore
import matplotlib.pyplot as plt
import tensorflow as tf  # type: ignore

from pumaguard.presets import Presets
from pumaguard.traininghistory import TrainingHistory
from pumaguard.utils import (
    create_datasets,
    initialize_tensorflow,
    organize_data,
)
from pumaguard.models.pretrained import pre_trained_model
from pumaguard.models.light import light_model
from pumaguard.models.light_2 import light_model_2


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
        type=bool,
        action='store_true',
    )
    return parser.parse_args()


def main():
    """
    The main entry point.
    """
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
