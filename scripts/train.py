"""
Train a model.
"""

import glob
import os
import shutil
import tempfile
import tensorflow as tf
import keras

# Use data from local directory
base_data_directory = os.path.join(os.path.dirname(__file__), '../data')
base_output_directory = os.path.join(os.path.dirname(__file__), '../models')

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

MODEL_FILE = f'{base_output_directory}/' \
    f'model_weights_{NOTEBOOK_NUMBER}_{MODEL_VERSION}' \
    f'_{image_dimensions[0]}_{image_dimensions[1]}.keras'
HISTORY_FILE = f'{base_output_directory}/' \
    f'model_history_{NOTEBOOK_NUMBER}_{MODEL_VERSION}' \
    f'_{image_dimensions[0]}_{image_dimensions[1]}.pickle'

print(f'Model file   {MODEL_FILE}')
print(f'History file {HISTORY_FILE}')

# Copy images to working directory on runtime


# Find image names in Google Drive
lion_images = []
for lion in lion_directories:
    lion_images += glob.glob(os.path.join(lion, '*JPG'))
no_lion_images = []
for no_lion in no_lion_directories:
    no_lion_images += glob.glob(os.path.join(no_lion, '*JPG'))

print(f'Found {len(lion_images)} images tagged as `lion`')
print(f'Found {len(no_lion_images)} images tagges as `no-lion`')
print(f'In total {len(lion_images) + len(no_lion_images)} images')

work_directory = tempfile.mkdtemp(prefix='pumaguard-work-')

shutil.rmtree(work_directory, ignore_errors=True)
os.makedirs(f'{work_directory}/lion')
os.makedirs(f'{work_directory}/no_lion')


def copy_images():
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


if MODEL_VERSION == 'pre-trained':
    COLOR_MODE = 'rgb'
else:
    COLOR_MODE = 'grayscale'
print(f'Using color_mode \'{COLOR_MODE}\'')

# Define augmentation layers which are used in some of the runs
augmentation_layers = [
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.01),
    keras.layers.RandomZoom(0.05),
    keras.layers.RandomBrightness((-0.1, 0.1)),
    keras.layers.RandomContrast(0.1),
    # tf.keras.layers.RandomCrop(200, 200),
    # tf.keras.layers.Rescaling(1./255),
]


def image_augmentation(image):
    """
    Use augmentation if `with_augmentation` is set to True
    """
    if WITH_AUGMENTATION:
        for layer in augmentation_layers:
            image = layer(image)
    return image


copy_images()

# Create datasets(training, validation)
training_dataset, validation_dataset = \
    keras.preprocessing.image_dataset_from_directory(
        work_directory,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='both',
        # Seed is always the same in order to ensure that we can reproduce the
        # same training session
        seed=123,
        shuffle=True,
        image_size=image_dimensions,
        color_mode=COLOR_MODE,
    )

training_dataset = training_dataset.map(
    lambda img, label: (image_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

print(f'image dimensions {image_dimensions}')
