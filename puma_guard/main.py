"""
PumaGuard
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
