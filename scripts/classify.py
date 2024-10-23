"""
This script classifies images.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import keras


# Use data from local directory
base_data_directory = os.path.realpath(os.path.join(
    os.path.dirname(__file__), '../data'))
base_output_directory = os.path.realpath(os.path.join(
    os.path.dirname(__file__), '../models'))

# Initialize Tensorflow

# Imports tensorflow into notebook, which has the Xception model defined inside
# it
print("Tensorflow version " + tf.__version__)

# Try different backends in the following order: TPU, GPU, CPU and use the
# first one available
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    distribution_strategy = tf.distribute.TPUStrategy(tpu)
    print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
except ValueError:
    print("WARNING: Not connected to a TPU runtime; Will try GPU")
    if tf.config.list_physical_devices('GPU'):
        distribution_strategy = tf.distribute.MirroredStrategy()
        print(f'Running on {len(tf.config.list_physical_devices("GPU"))} GPUs')
    else:
        print('WARNING: Not connected to TPU or GPU runtime; '
              'Will use CPU context')
        distribution_strategy = tf.distribute.get_strategy()


# Settings for the different rows in the table

# Set the notebook number to run.
NOTEBOOK_NUMBER = 6

# Load an existing model and its weights from disk (True) or create a fresh new
# model (False).
LOAD_MODEL_FROM_FILE = True

# Load previous training history from file (True).
LOAD_HISTORY_FROM_FILE = True

# How many epochs to train for.
EPOCHS = 300

# Default learning rate.
ALPHA = 1e-5

# No changes below this line.
if NOTEBOOK_NUMBER == 1:
    EPOCHS = 2_100
    image_dimensions = (128, 128)  # height, width
    WITH_AUGMENTATION = False
    BATCH_SIZE = 16
    MODEL_VERSION = "light"
    ALPHA = 1e-5
    lion_directories = [
        # f'{base_data_directory}/lion_1',
        f'{base_data_directory}/lion',
    ]
    no_lion_directories = [
        # f'{base_data_directory}/no_lion_1',
        f'{base_data_directory}/no_lion',
    ]
elif NOTEBOOK_NUMBER == 2:
    EPOCHS = 1_200
    image_dimensions = (256, 256)  # height, width
    WITH_AUGMENTATION = False
    BATCH_SIZE = 32
    MODEL_VERSION = "light"
    lion_directories = [
        # f'{base_data_directory}/lion_1',
        f'{base_data_directory}/lion',
    ]
    no_lion_directories = [
        # f'{base_data_directory}/no_lion_1',
        f'{base_data_directory}/no_lion',
    ]
elif NOTEBOOK_NUMBER == 3:
    EPOCHS = 900
    image_dimensions = (256, 256)  # height, width
    WITH_AUGMENTATION = True
    BATCH_SIZE = 32
    MODEL_VERSION = "light"
    lion_directories = [
        f'{base_data_directory}/lion',
    ]
    no_lion_directories = [
        f'{base_data_directory}/no_lion',
    ]
elif NOTEBOOK_NUMBER == 4:
    image_dimensions = (128, 128)  # height, width
    WITH_AUGMENTATION = False
    BATCH_SIZE = 16
    MODEL_VERSION = "pre-trained"
    lion_directories = [
        f'{base_data_directory}/lion_1',
    ]
    no_lion_directories = [
        f'{base_data_directory}/no_lion_1',
    ]
elif NOTEBOOK_NUMBER == 5:
    image_dimensions = (128, 128)  # height, width
    WITH_AUGMENTATION = False
    BATCH_SIZE = 16
    MODEL_VERSION = "pre-trained"
    lion_directories = [
        f'{base_data_directory}/lion',
    ]
    no_lion_directories = [
        f'{base_data_directory}/no_lion',
    ]
elif NOTEBOOK_NUMBER == 6:
    image_dimensions = (512, 512)  # height, width
    WITH_AUGMENTATION = False
    BATCH_SIZE = 16
    MODEL_VERSION = "pre-trained"
    lion_directories = [
        f'{base_data_directory}/lion',
        f'{base_data_directory}/cougar',
    ]
    no_lion_directories = [
        f'{base_data_directory}/no_lion',
        f'{base_data_directory}/nocougar',
    ]
else:
    raise ValueError(f'Unknown notebook {NOTEBOOK_NUMBER}')

model_file = (f'{base_output_directory}/model_weights_{NOTEBOOK_NUMBER}_'
              f'{MODEL_VERSION}_{image_dimensions[0]}_'
              f'{image_dimensions[1]}.keras')
history_file = (f'{base_output_directory}/model_history_{NOTEBOOK_NUMBER}_'
                f'{MODEL_VERSION}_{image_dimensions[0]}_'
                f'{image_dimensions[1]}.pickle')

print(f'Model file   {model_file}')
print(f'History file {history_file}')


# Define callbacks for training.
class StoreHistory(keras.callbacks.Callback):
    """
    Store the training history.
    """

    def __init__(self):
        super().__init__()
        self.history = {}
        self.number_epochs = 0
        history_file_exists = os.path.isfile(history_file)
        if history_file_exists and LOAD_HISTORY_FROM_FILE:
            print(f'Loading history from file {history_file}')
            with open(history_file, 'rb') as f:
                self.history = pickle.load(f)
                keys = list(self.history.keys())
                self.number_epochs = len(self.history[keys[0]])
                print(
                    f'Loaded history of {self.number_epochs} previous epochs')
                last_output = f'Epoch {self.number_epochs}: '
                for key in keys:
                    last_output += f'{key}: {self.history[key][-1]:.4f}'
                    if key != keys[-1]:
                        last_output += ' - '
                print(last_output)
        else:
            print(f'Creating new history file {history_file}')
        for key in ['duration', 'accuracy']:
            if key not in self.history:
                self.history[key] = []

    def on_train_begin(self, logs=None):
        keys = list(self.history.keys())
        if len(keys) == 0:
            self.number_epochs = 0
        else:
            self.number_epochs = len(self.history[keys[0]])
        print(
            f'Starting new training with {self.number_epochs} previous epochs')

    def on_epoch_end(self, epoch, logs=None):
        if 'batch_size' not in self.history:
            self.history['batch_size'] = []
        self.history['batch_size'].append(BATCH_SIZE)
        for key in logs:
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(logs[key])
        with open(history_file, 'wb') as f:
            pickle.dump(self.history, f)
            print(f'Epoch {epoch + self.number_epochs + 1} '
                  'history pickled and saved to file')


checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=model_file,
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

# Create / load training history


def get_best_epoch(history, key):
    """
    Get the best Epoch.
    """
    max_value = 0
    max_epoch = 0
    if key not in history.history or len(history.history[key]) == 0:
        return 0, 0, 0, 0, 0
    for epoch in range(len(history.history[key])):
        value = history.history[key][epoch]
        if value >= max_value:  # We want the last, best value
            max_value = value
            max_epoch = epoch
    return (history.history['accuracy'][max_epoch],
            history.history['val_accuracy'][max_epoch],
            history.history['loss'][max_epoch],
            history.history['val_loss'][max_epoch],
            max_epoch,
            )


full_history = StoreHistory()

best_accuracy, best_val_accuracy, best_loss, best_val_loss, best_epoch = \
    get_best_epoch(full_history, 'accuracy')
print(f'Total time {sum(full_history.history["duration"])} '
      f'for {len(full_history.history["accuracy"])} epochs')
print(f'Best epoch {best_epoch} - accuracy: {best_accuracy:.4f} - '
      f'val_accuracy: {best_val_accuracy:.4f} - loss: {best_loss:.4f} '
      f'- val_loss: {best_val_loss:.4f}')

# Define the two models.


def pre_trained_model():
    """
    Use the Xception model with imagenet weights as base model
    """
    base_model = keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(*image_dimensions, 3),
    )

    print(f'Number of layers in the base model: {len(base_model.layers)}')
    print(f'shape of output layer: {base_model.layers[-1].output_shape}')

    # We do not want to change the weights in the Xception model (imagenet
    # weights are frozen)
    base_model.trainable = False

    # Average pooling takes the 2,048 outputs of the Xeption model and brings
    # it into one output. The sigmoid layer makes sure that one output is
    # between 0-1. We will train all parameters in these last two layers
    return keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])


def light_model():
    """
    The light model does not run properly on a TPU runtime. The loss function
    results in `nan` after only one epoch. It does work on GPU runtimes though.
    """
    inputs = keras.Input(shape=(*image_dimensions, 1))

    # Entry block
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.Conv2D(128, 1, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(1, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = keras.layers.SeparableConv2D(1024, 1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dropout(0.1)(x)

    outputs = keras.layers.Dense(1, activation=None)(x)
    return keras.Model(inputs, outputs)

# Build model
#
# Prepares model so we can run it


with distribution_strategy.scope():
    model_file_exists = os.path.isfile(model_file)
    if LOAD_MODEL_FROM_FILE and model_file_exists:
        os.stat(model_file)
        print(f'Loading model from file {model_file}')
        model = keras.models.load_model(model_file)
        print('Loaded model from file')
    else:
        print('Creating new model')
        if MODEL_VERSION == "pre-trained":
            print('Creating new Xception model')
            model = pre_trained_model()
        elif MODEL_VERSION == "light":
            print('Creating new light model')
            model = light_model()
        else:
            raise ValueError(f'unknown model version {MODEL_VERSION}')

        if MODEL_VERSION == "pre-trained":
            model.build(input_shape=(None, *image_dimensions, 3))
        else:
            model.build(input_shape=(None, *image_dimensions, 1))

        print(f'Number of layers in the model: {len(model.layers)}')

# Compile model

with distribution_strategy.scope():
    if MODEL_VERSION == 'pre-trained':
        print('Compiling pre-trained model')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
    elif MODEL_VERSION == 'light':
        print('Compiling light model')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=ALPHA),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
        )
    else:
        raise ValueError(f'Unknown model version {MODEL_VERSION}')
    model.summary()


if MODEL_VERSION == 'pre-trained':
    COLOR_MODE = 'rgb'
else:
    COLOR_MODE = 'grayscale'
print(f'Using color_mode \'{COLOR_MODE}\'')


def classify_images():
    """
    Classify the images and print out the result.
    """
    verification_dataset = keras.preprocessing.image_dataset_from_directory(
        f'{base_data_directory}/stable/angle 4',
        batch_size=BATCH_SIZE,
        image_size=image_dimensions,
        shuffle=True,
        color_mode=COLOR_MODE,
    )

    all_predictions = []
    all_labels = []

    for images, labels in verification_dataset:
        predictions = model.predict(images)
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


def main():
    """
    Main entry point
    """
    classify_images()


if __name__ == "__main__":
    main()
