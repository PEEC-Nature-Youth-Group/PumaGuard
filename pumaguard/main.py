"""
PumaGuard
"""

import argparse
import logging
import sys

from pumaguard import (
    __VERSION__,
    classify,
    server,
    train,
)
from pumaguard.utils import (
    print_bash_completion,
)


def create_global_parser() -> argparse.ArgumentParser:
    """
    Shared arguments.
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        '--settings',
        help='Load presets from file',
        type=str,
    )
    global_parser.add_argument(
        '--version',
        action='version',
        version=__VERSION__,
    )
    global_parser.add_argument(
        '--completion',
        choices=['bash'],
        help='Print out bash completion script.',
    )
    global_parser.add_argument(
        '--notebook',
        help='The notebook number',
        type=int,
        default=1,
    )
    return global_parser


def main():
    """
    Main entry point.
    """
    logging.basicConfig(level=logging.INFO)

    global_args_parser = create_global_parser()
    parser = argparse.ArgumentParser(
        description='''The goal of this project is to accurately classify
                    images based on the presence of mountain lions. This can
                    have applications in wildlife monitoring, research, and
                    conservation efforts. The model is trained on a labeled
                    dataset and validated using a separate set of images.''',
        parents=[global_args_parser],
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Aavailable sub-commands',
    )
    train.configure_subparser(subparsers.add_parser(
        'train',
        help='Train a model',
        description='Train a model',
        parents=[global_args_parser],
    ))
    classify.configure_subparser(subparsers.add_parser(
        'classify',
        help='Classify images',
        description='Classify images using a particular model',
        parents=[global_args_parser],
    ))
    server.configure_subparser(subparsers.add_parser(
        'server',
        help='Run the classification server',
        parents=[global_args_parser],
    ))

    args = parser.parse_args()

    if args.completion:
        print_bash_completion(command=args.command, shell=args.completion)
        sys.exit(0)

    if args.command == 'train':
        train.main(args)
    elif args.command == 'server':
        server.main(args)
    elif args.command == 'classify':
        classify.main(args)
    else:
        parser.print_help()
