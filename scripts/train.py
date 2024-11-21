#!/usr/bin/env python3

"""
Convenience script to run the training.
"""

import sys
import os.path

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath(f'{os.path.dirname(__file__)}/..'))

from pumaguard.cmd.train import main

if __name__ == '__main__':
    sys.exit(main())
