"""
Test Presets class
"""

import os
import unittest

from pumaguard.presets import (
    BasePreset,
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

    def test_notebook_number_getter_default(self):
        """
        Test whether the default notebook number is correcct.
        """
        presets = Presets(1)
        self.assertEqual(presets.notebook_number, 1)

    def test_notebook_number_setter(self):
        """
        Test whether one can set a notebook number.
        """
        presets = Presets(1)
        with self.assertRaises(ValueError) as e:
            presets.notebook_number = 0
        self.assertEqual(str(e.exception),
                         'notebook can not be zero or negative (0)')


class TestBasePreset(unittest.TestCase):
    """
    Test the base class.
    """

    def setUp(self):
        self.base_preset = BasePreset()

    def test_image_dimensions_default(self):
        """
        Test the default value of image dimensions.
        """
        self.assertEqual(len(self.base_preset.image_dimensions), 2)
        self.assertEqual(self.base_preset.image_dimensions, (128, 128))

    def test_image_dimensions_failure(self):
        """
        Test various failures of image dimensions.
        """
        with self.assertRaises(TypeError) as e:
            self.base_preset.image_dimensions = 1
        self.assertEqual(str(e.exception),
                         'image dimensions needs to be a tuple')
        with self.assertRaises(ValueError) as e:
            self.base_preset.image_dimensions = (-1, 2)
        self.assertEqual(str(e.exception),
                         'image dimensions need to be positive')
