"""
Test server
"""

import unittest
from unittest.mock import patch
import sys

from pumaguard.cmd import server


class TestServer(unittest.TestCase):
    """
    Test pumaguard-server.
    """

    def test_parse_commandline(self):
        """
        Test whether we can parse command line arguments.
        """
        with patch.object(sys, 'argv', ['command', '--help']):
            with patch('sys.exit') as mock_exit:
                _ = server.parse_commandline()
                mock_exit.assert_called()
