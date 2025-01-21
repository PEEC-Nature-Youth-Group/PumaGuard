"""
PumaGuard
"""

import argparse
import logging
import sys

from pumaguard import (
    __VERSION__,
    train,
)
from pumaguard.utils import (
    print_bash_completion,
)


def main():
    """
    Main entry point.
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='''PumaGuard is a set of....'''
    )

    parser.add_argument(
        '--notebook',
        help='The notebook number',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--settings',
        help='Load presets from file',
        type=str,
    )
    parser.add_argument(
        '--version',
        action='version',
        version=__VERSION__,
    )
    parser.add_argument(
        '--completion',
        choices=['bash'],
        help='Print out bash completion script.',
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='sub commands',
    )

    train.configure_subparser(subparsers.add_parser('train'))

    # parser_server = subparsers.add_parser('server')
    # TODO Add subparser for server

    # parser_classify = subparsers.add_parser('classify')
    # TODO Add subparser for classify

    args = parser.parse_args()

    if args.completion:
        print_bash_completion(command=args.command, shell=args.completion)
        sys.exit(0)

    if args.command == 'train':
        train.main(args)
    elif args.command == 'server':
        # TODO Call server script
        pass
    elif args.command == 'classify':
        # TODO Call classify script
        pass
    else:
        parser.print_help()
