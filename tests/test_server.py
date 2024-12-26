"""
Test server.
"""

import unittest
from unittest.mock import patch
import sys

from pumaguard.server import (
    parse_commandline,
    FolderManager,
    FolderObserver,
)


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


class TestFolderManager(unittest.TestCase):
    """
    Unit tests for FolderManager class
    """

    def setUp(self):
        self.notebook = 1
        self.manager = FolderManager(self.notebook)

    @patch('pumaguard.server.FolderObserver')
    def test_register_folder(self, MockFolderObserver):  # pylint: disable=invalid-name
        """
        Test register folder.
        """
        folder = 'test_folder'
        self.manager.register_folder(folder)
        self.assertEqual(len(self.manager.observers), 1)
        MockFolderObserver.assert_called_with(folder, self.notebook)

    @patch.object(FolderObserver, 'start')
    def test_start_all(self, mock_start):
        """
        Test the start_all method.
        """
        folder = 'test_folder'
        self.manager.register_folder(folder)
        self.manager.start_all()
        mock_start.assert_called_once()

    @patch.object(FolderObserver, 'stop')
    def test_stop_all(self, mock_stop):
        """
        Test the stop_all method.
        """
        folder = 'test_folder'
        self.manager.register_folder(folder)
        self.manager.start_all()
        self.manager.stop_all()
        mock_stop.assert_called_once()
