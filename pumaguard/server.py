"""
Pumaguard server watches folders for new images and returns the probability
that the new images show pumas.

.. code-block:: shell

    usage: pumaguard-server [-h] [--debug] [--notebook NOTEBOOK] [--completion {bash}] [--watch-method {inotify,os}] [FOLDER ...]

    positional arguments:
      FOLDER                The folder(s) to watch. Can be used multiple times.

    options:
      -h, --help            show this help message and exit
      --debug               Debug the application
      --notebook NOTEBOOK   The notebook number
      --completion {bash}   Print out bash completion script.
      --watch-method {inotify,os}
                            What implementation (method) to use for watching the
                            folder. Linux on baremetal supports both methods. Linux
                            in WSL supports inotify on folders using ext4 but only
                            os on folders that are mounted from the Windows host.
"""  # pylint: disable=line-too-long

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from pumaguard.utils import classify_images

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
        nargs='*',
    )
    parser.add_argument(
        '--completion',
        choices=['bash'],
        help='Print out bash completion script.',
    )
    parser.add_argument(
        '--watch-method',
        help='''What implementation (method) to use for watching
        the folder. Linux on baremetal supports both methods. Linux
        in WSL supports inotify on folders using ext4 but only os
        on folders that are mounted from the Windows host. Defaults
        to "%(default)s"''',
        choices=['inotify', 'os'],
        default='os',
    )
    options = parser.parse_args()
    if options.completion:
        if options.completion == 'bash':
            print_bash_completion()
            sys.exit(0)
        else:
            raise ValueError(f'unknown completion {options.completion}')
    if not options.FOLDER:
        raise ValueError('missing FOLDER argument')
    return options


def print_bash_completion():
    """
    Print bash completion script.
    """
    print('''#!/bin/bash

_pumaguard_server_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-h --debug --help --notebook --watch-method --completion"

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
        --watch-method)
            opts="inotify os"
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

complete -F _pumaguard_server_completions pumaguard.pumaguard-server
complete -F _pumaguard_server_completions pumaguard-server
''')


class FolderObserver:
    """
    FolderObserver watches a folder for new files.
    """

    def __init__(self, folder: str, notebook: int, method: str):
        self.folder = folder
        self.notebook = notebook
        self.method = method
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
        if self.method == 'inotify':
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
                        args=(line.strip(),),
                    ).start()
        elif self.method == 'os':
            known_files = set(os.listdir(self.folder))
            while not self._stop_event.is_set():
                current_files = set(os.listdir(self.folder))
                new_files = current_files - known_files
                for new_file in new_files:
                    full_path = os.path.join(self.folder, new_file)
                    logger.info('New file detected: %s', full_path)
                    threading.Thread(
                        target=self._handle_new_file,
                        args=(full_path,),
                    ).start()
                known_files = current_files
                time.sleep(1)
        else:
            raise ValueError('FIXME: This method is not implemented')

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

            predictions = classify_images(self.notebook, workdir)
            logger.info('Chance of puma in %s: %.2f%%',
                        filepath, (1 - predictions[0]) * 100)


class FolderManager:
    """
    FolderManager manages the folders to observe.
    """

    def __init__(self, notebook: int):
        self.notebook = notebook
        self.observers: list[FolderObserver] = []

    def register_folder(self, folder: str, method: str):
        """
        Register a new folder for observation.

        Arguments:
            folder -- The path of the folder to watch.
        """
        observer = FolderObserver(folder, self.notebook, method)
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
        manager.register_folder(folder, options.watch_method)

    manager.start_all()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop_all()
        logger.info('Stopped watching folders.')
