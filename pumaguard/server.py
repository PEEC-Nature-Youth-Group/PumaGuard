"""
Pumaguard server watches folders for new images and returns the probability
that the new images show pumas.

usage: pumaguard-server [-h] [--debug] \
    [--notebook NOTEBOOK] FOLDER [FOLDER ...]

positional arguments:
  FOLDER               The folder(s) to watch. Can be used multiple times.

options:
  -h, --help           show this help message and exit
  --debug              Debug the application
  --notebook NOTEBOOK  The notebook number
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

import keras  # type: ignore

from pumaguard.utils import (
    create_model,
    initialize_tensorflow,
    Presets,
)

logger = logging.getLogger('PumaGuard-Server')


def initialize_logger():
    """
    Initializes and configures the logger for the application.

    This function sets up the logging configuration, including log level, log
    format, and log file handlers. It ensures that all log messages are
    properly formatted and directed to the appropriate output destinations.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)


def parse_commandline() -> argparse.Namespace:
    """
    Parses the command line arguments provided to the script.

    Returns:
        argparse.Namespace: An object containing the parsed command line
        arguments.
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
        'FOLDER',
        help='The folder(s) to watch. Can be used multiple times.',
        nargs='+',
    )
    return parser.parse_args()


class FolderObserver:
    """
    FolderObserver watches a folder for new files.
    """

    def __init__(self, folder: str, notebook: int):
        self.folder = folder
        self.notebook = notebook
        self._stop_event = threading.Event()

    def start(self):
        """
        start watching the folder.
        """
        self._stop_event.clear()
        threading.Thread(target=self._observe).start()

    def stop(self):
        """
        Stop watching the folder.
        """
        self._stop_event.set()

    def _observe(self):
        """
        Observe whether a new file is created in the folder.
        """
        with subprocess.Popen(
            ['inotifywait', '--monitor', '--event',
                'create', '--format', '%w%f', self.folder,],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            text=True,
        ) as process:
            for line in process.stdout:
                if self._stop_event.is_set():
                    process.terminate()
                    break
                logger.info('New file detected: %s', line.strip())
                threading.Thread(
                    target=self._handle_new_file,
                    args=(line.strip(),)).start()

    def _handle_new_file(self, filepath: str):
        """
        Handle the new file detected in the folder.

        Arguments:
            filepath -- The path of the new file.
        """
        logger.info('Handling new file: %s', filepath)
        with tempfile.TemporaryDirectory() as workdir:
            logger.info('working in folder %s', workdir)
            os.makedirs(os.path.join(workdir, 'lion'))
            os.makedirs(os.path.join(workdir, 'no-lion'))
            shutil.copy(filepath, os.path.join(workdir, 'lion'))

            presets = Presets(self.notebook)
            distribution_strategy = initialize_tensorflow()
            model = create_model(presets, distribution_strategy)
            if presets.model_version == 'pre-trained':
                color_model = 'rgb'
            else:
                color_model = 'grayscale'
            try:
                logger.info('creating dataaset')
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
                    logger.info('Chance of puma in %s: %.2f%%',
                                filepath, (1 - predictions) * 100)
            except ValueError as e:
                logger.error('unable to process file: %s', e)


class FolderManager:
    """
    FolderManager manages the folders to observe.
    """

    def __init__(self, notebook: int):
        self.notebook = notebook
        self.observers: list[FolderObserver] = []

    def register_folder(self, folder: str):
        """
        Register a new folder for observation.

        Arguments:
            folder -- The path of the folder to watch.
        """
        observer = FolderObserver(folder, self.notebook)
        self.observers.append(observer)
        logger.info('registered %s', folder)

    def start_all(self):
        """
        Start watching all registered folders.
        """
        logger.info('starting to watch folders')
        for observer in self.observers:
            observer.start()

    def stop_all(self):
        """
        Stop watching all registered folders.
        """
        logger.info('stopping to watch all folders')
        for observer in self.observers:
            observer.stop()


def main():
    """
    Main entry point.
    """

    initialize_logger()
    options = parse_commandline()
    if options.debug:
        logger.setLevel(logging.DEBUG)

    manager = FolderManager(options.notebook)
    for folder in options.FOLDER:
        manager.register_folder(folder)

    manager.start_all()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop_all()
        logger.info('Stopped watching folders.')
