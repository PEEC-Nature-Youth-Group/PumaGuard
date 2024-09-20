"""
Test main.
"""

import unittest
from unittest.mock import patch
import sys

from pumaguard import main


class TestMain(unittest.TestCase):
    """
    Test Main.
    """

    def test_parse_commandline(self):
        """
        Test whether we can parse command line arguments.
        """
        with patch.object(sys, 'argv', ['command', '--help']):
            with patch('sys.exit') as mock_exit:
                _ = main.parse_commandline()
                mock_exit.assert_called()
