"""
Some utility functions.
"""

import datetime
import glob
import hashlib
import logging
import os
import shutil

import keras  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
from PIL import (
    Image,
)

from pumaguard.models.light import (
    light_model,
)
from pumaguard.models.light_2 import (
    light_model_2,
)
from pumaguard.models.pretrained import (
    pre_trained_model,
)
from pumaguard.presets import (
    Presets,
)

logger = logging.getLogger('PumaGuard-Server')


def initialize_tensorflow() -> tf.distribute.Strategy:
    """
    Initialize Tensorflow on available hardware.

    Try different backends in the following order: TPU, GPU, CPU and use the
    first one available.

    Returns:
        tf.distribute.Strategy: The distribution strategy object after
        initialization.
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


def organize_data(presets: Presets, work_directory: str):
    """
    Organizes the data and splits it into training and validation datasets.
    """
    lion_images = []
    for lion in presets.lion_directories:
        lion_images += glob.glob(os.path.join(lion, '*JPG'))
    no_lion_images = []
    for no_lion in presets.no_lion_directories:
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


def image_augmentation(image, with_augmentation: bool, augmentation_layers):
    """
    Use augmentation if `with_augmentation` is set to True
    """
    if with_augmentation:
        for layer in augmentation_layers:
            image = layer(image)
    return image


def create_datasets(presets: Presets, work_directory: str, color_mode: str):
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
            batch_size=presets.batch_size,
            validation_split=0.2,
            subset='both',
            # Seed is always the same in order to ensure that we can reproduce
            # the same training session
            seed=123,
            shuffle=True,
            image_size=presets.image_dimensions,
            color_mode=color_mode,
        )

    training_dataset = training_dataset.map(
        lambda img, label: (image_augmentation(
            image=img,
            with_augmentation=presets.with_augmentation,
            augmentation_layers=augmentation_layers), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return training_dataset, validation_dataset


def get_md5(filepath: str) -> str:
    """
    Compute the MD5 hash for a file.
    """
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def get_sha256(filepath: str) -> str:
    """
    Compute the SHA-256 hash for a file.
    """
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def create_model(presets: Presets,
                 distribution_strategy: tf.distribute.Strategy) \
        -> keras.src.Model:
    """
    Create the model.
    """
    with distribution_strategy.scope():
        logger.info('looking for model at %s', presets.model_file)
        model_file_exists = os.path.isfile(presets.model_file)
        if presets.load_model_from_file and model_file_exists:
            os.stat(presets.model_file)
            logger.debug('loading model from file %s', presets.model_file)
            model = keras.models.load_model(presets.model_file)
            logger.debug('loaded model from file')
            logger.info('model version %s', get_md5(presets.model_file))
        else:
            logger.info('could not find model; creating new model')
            if presets.model_version == "pre-trained":
                logger.debug('Creating new Xception model')
                model = pre_trained_model(presets)
                logger.debug('Building pre-trained model')
                model.build(input_shape=(None, *presets.image_dimensions, 3))
                logger.debug('Compiling pre-trained model')
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                )
            elif presets.model_version == "light":
                logger.info('Creating new light model')
                model = light_model(presets)
                logger.debug('Building light model')
                model.build(input_shape=(None, *presets.image_dimensions, 1))
                logger.debug('Compiling light model')
                model.compile(
                    optimizer=keras.optimizers.Adam(
                        learning_rate=presets.alpha),
                    loss=keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
                )
            elif presets.model_version == 'light-2':
                logger.debug('Creating new light-2 model')
                model = light_model_2(presets)
                logger.debug('Building light-2 model')
                model.build(input_shape=(None, *presets.image_dimensions, 1))
                logger.debug('Compiling light model')
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                )
            else:
                raise ValueError(
                    f'unknown model version {presets.model_version}')

        logger.info('Number of layers in the model: %d', len(model.layers))
        model.summary()

    return model


class Model():
    """
    The Model used.
    """

    _instance = None
    _initialized = False

    def __new__(cls, presets: Presets):
        """
        Create a new model.
        """
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
        return cls._instance

    def __init__(self, presets: Presets):
        if not self._initialized:
            self._presets = presets
            self._distribution_strategy = initialize_tensorflow()
            self._model = create_model(
                self._presets, self._distribution_strategy)
            self._initialized = True

    def get_model(self) -> keras.src.Model:
        """
        Get the model.
        """
        return self._model


def classify_directory(notebook: int, workdir: str) -> list[float]:
    """
    Classify images in a workdir and return the probabilities.
    """
    presets = Presets(notebook)
    model = Model(presets).get_model()
    if presets.model_version == 'pre-trained':
        color_model = 'rgb'
    else:
        color_model = 'grayscale'
    try:
        logger.info('creating dataset')
        verification_dataset = \
            keras.preprocessing.image_dataset_from_directory(
                workdir,
                label_mode=None,
                batch_size=presets.batch_size,
                image_size=presets.image_dimensions,
                color_mode=color_model,
            )
        logger.info('classifying images')
        for images in verification_dataset:
            logger.info('working on batch')
            predictions = model.predict(images)
        return predictions[0].tolist()
    except ValueError as e:
        logger.error('unable to process file: %s', e)
    return []


def classify_image(presets: Presets, model: keras.Model,
                   image_path: str) -> float:
    """
    Classify the image and print out the result.

    Args:
        presets (Presets): An instance of the Presets class containing image
        processing settings.

        model (keras.Model): A pre-trained Keras model used for image
        classification.

        image_path (str): The file path to the image to be classified.

    Returns:
        float: The classification result as a float value.

    Prints:
        The color mode being used, the image being classified, and the time
        taken for classification.
    """
    logger.debug('Using color_mode "%s"', presets.color_mode)
    logger.debug('Classifying image %s', image_path)

    if presets.color_mode == 'rgb':
        img = Image.open(image_path).convert('RGB')
    elif presets.color_mode == 'grayscale':
        img = Image.open(image_path).convert('L')
    else:
        raise ValueError(f'unknown color mode {presets.color_mode}')

    start_time = datetime.datetime.now()

    img = img.resize(presets.image_dimensions)
    img_array = np.array(img)
    # img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if presets.color_mode == 'grayscale':
        img_array = np.expand_dims(img_array, axis=-1)

    prediction = model.predict(img_array)

    end_time = datetime.datetime.now()

    logger.debug('Classification took %.2f seconds',
                 get_duration(start_time, end_time))

    return prediction[0][0]


def print_bash_completion(completions_file: str):
    """
    Print bash completion script.
    """
    completions_file = os.path.join(os.path.dirname(
        __file__), 'completions', completions_file)
    with open(completions_file, encoding='utf-8') as fd:
        print(fd.read())
