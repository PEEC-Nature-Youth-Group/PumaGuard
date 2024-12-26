"""
Test server.
"""

import unittest
from unittest.mock import patch
import sys

from pumaguard.server import parse_commandline


class TestFolderObserver(unittest.TestCase):
    """
    Unit tests for server
    """


class TestParseCommandline(unittest.TestCase):
    """
    Unit tests for parse_commandline function
    """

    @patch.object(sys, 'argv', ['pumaguard-server', '--debug',
                                '--notebook', '2', 'folder1', 'folder2'])
    def test_parse_commandline_with_all_arguments(self):
        """
        Test with all arguments.
        """
        options = parse_commandline()
        self.assertTrue(options.debug)
        self.assertEqual(options.notebook, 2)
        self.assertEqual(options.FOLDER, ['folder1', 'folder2'])

    @patch.object(sys, 'argv', ['pumaguard-server', 'folder1'])
    def test_parse_commandline_with_minimal_arguments(self):
        """
        Test with minimal arguments.
        """
        options = parse_commandline()
        self.assertFalse(options.debug)
        self.assertEqual(options.notebook, 1)
        self.assertEqual(options.FOLDER, ['folder1'])

    @patch.object(sys, 'argv', ['pumaguard-server', '--completion', 'bash'])
    def test_parse_commandline_with_completion(self):
        """
        Test completions.
        """
        with self.assertRaises(SystemExit):
            parse_commandline()

    @patch.object(sys, 'argv', ['pumaguard-server'])
    def test_parse_commandline_missing_folder(self):
        """
        Test missing folder.
        """
        with self.assertRaises(ValueError):
            parse_commandline()
