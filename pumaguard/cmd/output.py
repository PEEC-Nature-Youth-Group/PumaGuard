"""
PumaGuard Output Unit

This script will control the output unit and activate speakers or lights.
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
