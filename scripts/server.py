#!/usr/bin/env python3

"""
Convenience script to run the server.
"""

import os.path
import sys

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath(f'{os.path.dirname(__file__)}/..'))

from pumaguard.main import (
    main,
)

if __name__ == '__main__':
    sys.argv = [
        sys.argv[0], 'server', *sys.argv[1:],
    ]
    sys.exit(main())
