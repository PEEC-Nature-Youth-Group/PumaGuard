"""
PumaGuard Server

This script will run the machine learning model and receive images from the
trailcams for classification. It will contact the speakers and/or lights if a
puma was identified.
"""

import argparse


def parse_commandline() -> argparse.Namespace:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    """
    Entry point.
    """
    _ = parse_commandline()
    print('Hello World!')
