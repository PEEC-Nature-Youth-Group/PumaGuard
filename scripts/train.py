"""
Train a model.
"""

from datetime import datetime
import glob
import os
import pickle
import shutil
import tempfile
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


class TrainingHistory(keras.callbacks.Callback):
    """
    This class stores the training history
    """

    def __init__(self):
        super().__init__()
        self.history = {}
        self.number_epochs = 0
        history_file_exists = os.path.isfile(HISTORY_FILE)
        if history_file_exists and LOAD_HISTORY_FROM_FILE:
            print(f'Loading history from file {HISTORY_FILE}')
            with open(HISTORY_FILE, 'rb') as f:
                self.history = pickle.load(f)
                keys = list(self.history.keys())
                self.number_epochs = len(self.history[keys[0]])
                print(f'Loaded history of {self.number_epochs} '
                      'previous epochs')
                last_output = f'Epoch {self.number_epochs}: '
                for key in keys:
                    last_output += f'{key}: {self.history[key][-1]:.4f}'
                    if key != keys[-1]:
                        last_output += ' - '
                print(last_output)
        else:
            print(f'Creating new history file {HISTORY_FILE}')
        for key in ['duration', 'accuracy']:
            if key not in self.history:
                self.history[key] = []

    def on_train_begin(self, logs=None):
        keys = list(self.history.keys())
        if len(keys) == 0:
            self.number_epochs = 0
        else:
            self.number_epochs = len(self.history[keys[0]])
        print(f'Starting new training with {self.number_epochs} '
              'previous epochs')

    def on_epoch_end(self, epoch, logs=None):
        if 'batch_size' not in self.history:
            self.history['batch_size'] = []
        self.history['batch_size'].append(BATCH_SIZE)
        for key in logs:
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(logs[key])
        with open(HISTORY_FILE, 'wb') as f:
            pickle.dump(self.history, f)
            print(f'Epoch {epoch + self.number_epochs + 1}'
                  'history pickled and saved to file')

    def get_best_epoch(self, key):
        """
        get_best_epoch Get the best epoch so far.

        Arguments:
            history -- _description_
            key -- _description_

        Returns:
            _description_
        """
        max_value = 0
        max_epoch = 0
        if key not in self.history or \
                len(self.history[key]) == 0:
            return 0, 0, 0, 0, 0
        for epoch in range(len(self.history[key])):
            value = self.history[key][epoch]
            if value >= max_value:  # We want the last, best value
                max_value = value
                max_epoch = epoch
        return (self.history['accuracy'][max_epoch],
                self.history['val_accuracy'][max_epoch],
                self.history['loss'][max_epoch],
                self.history['val_loss'][max_epoch],
                max_epoch,
                )


def intialize_tensorflow() -> tf.distribute.Strategy:
    """
    Initialize Tensorflow on available hardware.

    Try different backends in the following order: TPU, GPU, CPU and use the
    first one available
    """
    print("Tensorflow version " + tf.__version__)
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
        return tf.distribute.TPUStrategy(tpu)
    except ValueError:
        print("WARNING: Not connected to a TPU runtime; Will try GPU")
        if tf.config.list_physical_devices('GPU'):
            print('Running on '
                  f'{len(tf.config.list_physical_devices("GPU"))} GPUs')
            return tf.distribute.MirroredStrategy()
        print('WARNING: Not connected to TPU or GPU runtime; '
              'Will use CPU context')
        return tf.distribute.get_strategy()


def copy_images(work_directory, lion_images, no_lion_images):
    """
    Copy images to work directory.
    """
    print(f'Copying images to working directory '
          f'{os.path.realpath(work_directory)}')
    for image in lion_images:
        shutil.copy(image, f'{work_directory}/lion')
    for image in no_lion_images:
        shutil.copy(image, f'{work_directory}/no_lion')
    print('Copied all images')


def image_augmentation(image, augmentation_layers):
    """
    Use augmentation if `with_augmentation` is set to True
    """
    if WITH_AUGMENTATION:
        for layer in augmentation_layers:
            image = layer(image)
    return image


def pre_trained_model():
    """
    The pre-trained model (Xception).

    Returns:
        The model.
    """
    # Use the Xception model with imagenet weights as base model
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
    Define the "light model" which is loosely based on the Xception model and
    constructs a CNN.

    Note, the light model does not run properly on a TPU runtime. The loss
    function results in `nan` after only one epoch. It does work on GPU
    runtimes though.
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


def organize_data(work_directory):
    """
    Organizes the data and splits it into training and validation datasets.
    """
    lion_images = []
    for lion in lion_directories:
        lion_images += glob.glob(os.path.join(lion, '*JPG'))
    no_lion_images = []
    for no_lion in no_lion_directories:
        no_lion_images += glob.glob(os.path.join(no_lion, '*JPG'))

    print(f'Found {len(lion_images)} images tagged as `lion`')
    print(f'Found {len(no_lion_images)} images tagged as `no-lion`')
    print(f'In total {len(lion_images) + len(no_lion_images)} images')

    shutil.rmtree(work_directory, ignore_errors=True)
    os.makedirs(f'{work_directory}/lion')
    os.makedirs(f'{work_directory}/no_lion')

    copy_images(work_directory=work_directory,
                lion_images=lion_images,
                no_lion_images=no_lion_images)


def create_datasets(work_directory, color_mode):
    """
    Create the training and validation datasets.
    """
    # Define augmentation layers which are used in some of the runs
    augmentation_layers = [
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(0.01),
        keras.layers.RandomZoom(0.05),
        keras.layers.RandomBrightness((-0.1, 0.1)),
        keras.layers.RandomContrast(0.1),
        # keras.layers.RandomCrop(200, 200),
        # keras.layers.Rescaling(1./255),
    ]

    # Create datasets(training, validation)
    training_dataset, validation_dataset = \
        keras.preprocessing.image_dataset_from_directory(
            work_directory,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            subset='both',
            # Seed is always the same in order to ensure that we can reproduce
            # the same training session
            seed=123,
            shuffle=True,
            image_size=image_dimensions,
            color_mode=color_mode,
        )

    training_dataset = training_dataset.map(
        lambda img, label: (image_augmentation(
            image=img, augmentation_layers=augmentation_layers), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return training_dataset, validation_dataset


def create_model(distribution_strategy):
    """
    Create the model.
    """
    with distribution_strategy.scope():
        model_file_exists = os.path.isfile(MODEL_FILE)
        if LOAD_MODEL_FROM_FILE and model_file_exists:
            os.stat(MODEL_FILE)
            print(f'Loading model from file {MODEL_FILE}')
            model = keras.models.load_model(MODEL_FILE)
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


def train_model(training_dataset, validation_dataset, full_history, model):
    """
    Train the model.
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=MODEL_FILE,
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

    start_time = datetime.now()
    print(start_time)
    model.fit(
        training_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=[
            checkpoint,
            reduce_learning_rate,
            full_history,
        ]
    )
    end_time = datetime.now()
    print(end_time)

    duration = (end_time - start_time).total_seconds()
    print(f'This run took {duration} seconds')

    if 'duration' not in full_history.history:
        full_history.history['duration'] = []
    full_history.history['duration'].append(duration)

    print(f'total time {sum(full_history.history["duration"])} '
          f'for {len(full_history.history["accuracy"])} epochs')


def main():
    """
    The main entry point.
    """
    print(f'Model file   {MODEL_FILE}')
    print(f'History file {HISTORY_FILE}')

    work_directory = tempfile.mkdtemp(prefix='pumaguard-work-')
    organize_data(work_directory=work_directory)

    if MODEL_VERSION == 'pre-trained':
        color_mode = 'rgb'
    else:
        color_mode = 'grayscale'
    print(f'Using color_mode \'{color_mode}\'')

    print(f'image dimensions {image_dimensions}')

    training_dataset, validation_dataset = create_datasets(work_directory,
                                                           color_mode)

    full_history = TrainingHistory()

    best_accuracy, best_val_accuracy, best_loss, best_val_loss, \
        best_epoch = full_history.get_best_epoch('accuracy')
    print(f'Total time {sum(full_history.history["duration"])} '
          f'for {len(full_history.history["accuracy"])} epochs')
    print(f'Best epoch {best_epoch} - accuracy: {best_accuracy:.4f}'
          f'- val_accuracy: {best_val_accuracy:.4f} - loss: '
          f'{best_loss:.4f} - val_loss: {best_val_loss:.4f}')

    distribution_strategy = intialize_tensorflow()
    model = create_model(distribution_strategy)
    train_model(training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                model=model,
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


base_data_directory = os.path.join(os.path.dirname(__file__), '../data')
base_output_directory = os.path.join(os.path.dirname(__file__), '../models')

# Set the notebook number to run.
NOTEBOOK_NUMBER = 1

# Load an existing model and its weights from disk (True) or create a fresh new
# model (False).
LOAD_MODEL_FROM_FILE = True

# Load previous training history from file (True).
LOAD_HISTORY_FROM_FILE = True

# How many epochs to train for.
EPOCHS = 300

# Default step size.
ALPHA = 1e-5

# No changes below this line.
if NOTEBOOK_NUMBER == 1:
    EPOCHS = 1
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

MODEL_FILE = f'{base_output_directory}/' \
    f'model_weights_{NOTEBOOK_NUMBER}_{MODEL_VERSION}' \
    f'_{image_dimensions[0]}_{image_dimensions[1]}.keras'
HISTORY_FILE = f'{base_output_directory}/' \
    f'model_history_{NOTEBOOK_NUMBER}_{MODEL_VERSION}' \
    f'_{image_dimensions[0]}_{image_dimensions[1]}.pickle'


if __name__ == '__main__':
    main()
