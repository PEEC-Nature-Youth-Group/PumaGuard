"""
PumaGuard
"""

import argparse
import logging
import os
import sys

from pumaguard import (
    __VERSION__,
    classify,
    server,
    train,
    verify,
)
from pumaguard.presets import (
    Preset,
)
from pumaguard.utils import (
    print_bash_completion,
)


def create_global_parser() -> argparse.ArgumentParser:
    """
    Shared arguments.
    """
    data_path = os.getenv(
        'PUMAGUARD_DATA_PATH',
        default=os.path.join(os.path.dirname(__file__), '../data'))
    model_path = os.getenv(
        'PUMAGUARD_MODEL_PATH',
        default=os.path.join(os.path.dirname(__file__), '../models'))

    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        '--settings',
        help='Load presets from file',
        type=str,
    )
    global_parser.add_argument(
        '--debug',
        help='Debug the application',
        action='store_true',
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
        '--model-path',
        help='Where the models are stored (default = %(default)s)',
        type=str,
        default=model_path,
    )
    global_parser.add_argument(
        '--data-path',
        help=('Where the image data for training are '
              'stored (default = %(default)s)'),
        type=str,
        default=data_path,
    )
    return global_parser


def main():
    """
    Main entry point.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('PumaGuard')

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
        description='Train a model and get the model weights.',
        parents=[global_args_parser],
    ))
    classify.configure_subparser(subparsers.add_parser(
        'classify',
        help='Classify images',
        description='Classify images using a particular model.',
        parents=[global_args_parser],
    ))
    server.configure_subparser(subparsers.add_parser(
        'server',
        help='Run the classification server',
        description='''Run the classification server. The server will monitor
                    folders and classify any new image added to those
                    folders.''',
        parents=[global_args_parser],
    ))
    verify.configure_subparser(subparsers.add_parser(
        'verify',
        help='Verify a model',
        description='Verifies a model using a standard set of images.',
        parents=[global_args_parser],
    ))

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.completion:
        print_bash_completion(command=args.command, shell=args.completion)
        sys.exit(0)

    presets = Preset()
    presets.load(args.settings)

    model_path = args.model_path if args.model_path \
        else os.getenv('PUMAGUARD_MODEL_PATH', default=None)
    if model_path is not None:
        logger.debug('setting model path to %s', model_path)
        presets.base_output_directory = model_path

    data_path = args.data_path if args.data_path \
        else os.getenv('PUMAGUARD_DATA_PATH', default=None)
    if data_path is not None:
        logger.debug('setting data path to %s', data_path)
        presets.base_data_directory = data_path

    if args.command == 'train':
        train.main(args, presets)
    elif args.command == 'server':
        server.main(args, presets)
    elif args.command == 'classify':
        classify.main(args, presets)
    elif args.command == 'verify':
        verify.main(presets)
    else:
        parser.print_help()
