"""
Pick random files to generate training set.
"""

import argparse
import os
import random
import shutil


def pick_files(directories: list[str], num_files: int, destination: str):
    """
    Pick num_files random files from directories and copy files to destination.

    Args:
        directories (list[str]): A list of directories containing files
        num_files (int): The number of files to copy
        destination (str): The destination directory

    Raises:
        FileNotFoundError: A directory was not found
    """
    os.makedirs(destination, exist_ok=True)
    files = []
    for directory in directories:
        try:
            files.extend([os.path.join(directory, f)
                          for f in os.listdir(directory)])
        except FileNotFoundError:
            print(f'could not open directory {directory}')
            raise

    random.shuffle(files)
    selected_files = files[:num_files]
    for file in selected_files:
        shutil.copy2(file, destination)


def parse_commandline() -> argparse.Namespace:
    """
    Parse the commandline.

    Returns:
        argparse.Namespace: The parsed options
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-files",
        metavar="N",
        help="Select and copy N files",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--source",
        metavar="PATH",
        help="Select files from PATH",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--destination",
        metavar="PATH",
        help="Copy the selected files to PATH",
        type=str,
    )
    return parser.parse_args()


def main():
    """
    The entry point.
    """

    options = parse_commandline()
    pick_files(options.source, options.num_files, options.destination)
