"""
PumaGuard Trailcam Unit

This script will monitor the trailcam attached to it and send images to the
server unit.
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
