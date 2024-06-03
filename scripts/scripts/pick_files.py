import argparse
import os
import random
import shutil


def pick_files(directories, num_files, destination):
    os.makedirs(destination, exist_ok=True)
    files = []
    for directory in directories:
        try:
            files.extend([os.path.join(directory, f)
                          for f in os.listdir(directory)])
        except FileNotFoundError as e:
            raise e

    random.shuffle(files)
    selected_files = files[:num_files]
    for file in selected_files:
        shutil.copy2(file, destination)


def parse_commandline():
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
    options = parse_commandline()
    pick_files(options.source, options.num_files, options.destination)
