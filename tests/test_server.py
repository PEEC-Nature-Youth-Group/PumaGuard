"""
Test server.
"""

import unittest
from unittest.mock import patch, MagicMock

import io
import os
import sys

from pumaguard.server import (
    parse_commandline,
    FolderManager,
    FolderObserver,
)


class TestFolderObserver(unittest.TestCase):
    """
    Unit tests for FolderObserver class
    """

    def setUp(self):
        self.folder = 'test_folder'
        self.notebook = 1
        self.observer = FolderObserver(self.folder, self.notebook, 'inotify')

    @patch('pumaguard.server.subprocess.Popen')
    def test_observe_new_file(self, MockPopen):  # pylint: disable=invalid-name
        """
        Test observing a new file.
        """
        mock_process = MagicMock()
        mock_process.stdout = iter(['test_folder/new_file.jpg\n'])
        MockPopen.return_value.__enter__.return_value = mock_process

        with patch.object(self.observer, '_handle_new_file') \
                as mock_handle_new_file:
            self.observer._observe()  # pylint: disable=protected-access
            mock_handle_new_file.assert_called_once_with(
                'test_folder/new_file.jpg')

    @patch('pumaguard.server.threading.Thread')
    def test_start(self, MockThread):  # pylint: disable=invalid-name
        """
        Test starting the observer.
        """
        self.observer.start()
        MockThread.assert_called_once_with(
            target=self.observer._observe)  # pylint: disable=protected-access
        MockThread.return_value.start.assert_called_once()

    def test_stop(self):
        """
        Test stopping the observer.
        """
        self.observer._stop_event = MagicMock()  # pylint: disable=protected-access
        self.observer.stop()
        self.observer._stop_event.set.assert_called_once()  # pylint: disable=protected-access

    @patch('pumaguard.server.shutil.copy')
    @patch('pumaguard.server.tempfile.TemporaryDirectory')
    @patch('pumaguard.server.classify_images')
    def test_handle_new_file(self, mock_classify_images,
                             MockTemporaryDirectory, mock_copy):  # pylint: disable=invalid-name
        """
        Test handling a new file.
        """
        mock_classify_images.return_value = [0.2]
        mock_temp_dir = MagicMock()
        MockTemporaryDirectory.return_value.__enter__.return_value = \
            mock_temp_dir

        self.observer._handle_new_file(  # pylint: disable=protected-access
            'test_folder/new_file.jpg')

        mock_copy.assert_called_once_with(
            'test_folder/new_file.jpg', os.path.join(mock_temp_dir, 'lion'))
        mock_classify_images.assert_called_once_with(
            self.notebook, mock_temp_dir)
        self.assertEqual(mock_classify_images.return_value, [0.2])


class TestParseCommandline(unittest.TestCase):
    """
    Unit tests for parse_commandline function
    """

    @patch.object(sys, 'argv', ['pumaguard-server', '--debug',
                                '--notebook', '2',
                                '--watch-method', 'inotify',
                                'folder1', 'folder2'])
    def test_parse_commandline_with_all_arguments(self):
        """
        Test with all arguments.
        """
        options = parse_commandline()
        self.assertTrue(options.debug)
        self.assertEqual(options.notebook, 2)
        self.assertEqual(options.FOLDER, ['folder1', 'folder2'])
        self.assertEqual(options.watch_method, 'inotify')

    @patch.object(sys, 'argv', ['pumaguard-server', 'folder1'])
    def test_parse_commandline_with_minimal_arguments(self):
        """
        Test with minimal arguments.
        """
        options = parse_commandline()
        self.assertFalse(options.debug)
        self.assertEqual(options.notebook, 1)
        self.assertEqual(options.FOLDER, ['folder1'])
        self.assertEqual(options.watch_method, 'os')

    @patch.object(sys, 'argv', ['pumaguard-server', '--completion',
                                'bash', 'FOLDER'])
    @patch('sys.exit')
    def test_parse_commandline_with_completion(self, mock_exit):
        """
        Test completions.
        """
        with patch('sys.stdout', new=io.StringIO()) as my_out:
            parse_commandline()
            result = my_out.getvalue()
            self.assertIn('complete -F', result)
        mock_exit.assert_called_once()

    @patch.object(sys, 'argv', ['pumaguard-server', '--completion',
                                'bash', 'FOLDER'])
    @patch('sys.exit')
    @patch('pumaguard.server.print_bash_completion')
    def test_parse_commandline_with_completion_2(self, mock_print, mock_exit):
        """
        Test completions.
        """
        parse_commandline()
        mock_exit.assert_called_once()
        mock_print.assert_called_once()

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
        self.manager.register_folder(folder, 'inotify')
        self.assertEqual(len(self.manager.observers), 1)
        MockFolderObserver.assert_called_with(folder, self.notebook, 'inotify')

    @patch.object(FolderObserver, 'start')
    def test_start_all(self, mock_start):
        """
        Test the start_all method.
        """
        folder = 'test_folder'
        self.manager.register_folder(folder, 'inotify')
        self.manager.start_all()
        mock_start.assert_called_once()

    @patch.object(FolderObserver, 'stop')
    def test_stop_all(self, mock_stop):
        """
        Test the stop_all method.
        """
        folder = 'test_folder'
        self.manager.register_folder(folder, 'inotify')
        self.manager.start_all()
        self.manager.stop_all()
        mock_stop.assert_called_once()
