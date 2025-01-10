"""
Test Presets class
"""

import os
import unittest

from pumaguard.presets import (
    Presets,
)


class TestPresets(unittest.TestCase):
    """
    Test Presets class.
    """

    def setUp(self):
        self.presets = Presets(1)

    def test_lion_directories_getter_default(self):
        """
        Test the default lion_directories for a known notebook_number.
        """
        self.assertIn(os.path.join(self.presets.base_data_directory, 'lion'),
                      self.presets.lion_directories)

    def test_lion_directories_getter_base(self):
        """
        Test whether changing the base data directory changes the lion
        directory.
        """
        presets = Presets(1)
        self.assertIn(os.path.join(presets.base_data_directory, 'lion'),
                      presets.lion_directories)
        presets.base_data_directory = '/tmp'
        self.assertIn(os.path.join('/tmp', 'lion'), presets.lion_directories)

    def test_lion_directories_setter(self):
        """
        Test setting custom lion_directories.
        """
        custom_lions = ['new_lion', 'another_lion']
        self.presets.lion_directories = custom_lions
        expected_paths = [
            os.path.join(self.presets.base_data_directory, lion)
            for lion in custom_lions
        ]
        self.assertListEqual(self.presets.lion_directories, expected_paths)

    def test_no_lion_directories_getter_default(self):
        """
        Test the default no_lion_directories for a known notebook_number.
        """
        self.assertIn(
                os.path.join(self.presets.base_data_directory, 'no_lion'),
                self.presets.no_lion_directories)

    def test_no_lion_directories_getter_base(self):
        """
        Test whether changing the base data directory changes the lion
        directory.
        """
        presets = Presets(1)
        self.assertIn(os.path.join(presets.base_data_directory, 'no_lion'),
                      presets.no_lion_directories)
        presets.base_data_directory = '/tmp'
        self.assertIn(os.path.join('/tmp', 'no_lion'),
                      presets.no_lion_directories)

    def test_no_lion_directories_setter(self):
        """
        Test setting custom lion_directories.
        """
        custom_no_lions = ['new_no_lion', 'another_no_lion']
        self.presets.no_lion_directories = custom_no_lions
        expected_paths = [
            os.path.join(self.presets.base_data_directory, no_lion)
            for no_lion in custom_no_lions
        ]
        self.assertListEqual(self.presets.no_lion_directories, expected_paths)
