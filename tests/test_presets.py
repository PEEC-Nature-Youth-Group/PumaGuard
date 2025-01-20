"""
Test Presets class
"""

import os
import unittest
from unittest.mock import (
    mock_open,
    patch,
)

from pumaguard.presets import (
    BasePreset,
    Preset01,
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

    @patch('builtins.open', new_callable=mock_open, read_data='''
notebook-number: 10
epochs: 2400
image-dimensions: [128, 128]
with-augmentation: True
batch-size: 2
model-version: light-test
alpha: 1e-3
base-data-directory: /path/to/data
base-output-directory: /path/to/output
lion-directories:
    - lion
no-lion-directories:
    - no_lion
''')
    def test_load(self, mock_file):  # pylint: disable=unused-argument
        """
        Test loading settings from file.
        """
        self.base_preset.load('/fake/path/to/settings.yaml')
        self.assertEqual(self.base_preset.notebook_number, 10)
        self.assertEqual(self.base_preset.epochs, 2400)
        self.assertEqual(self.base_preset.image_dimensions, (128, 128))
        self.assertEqual(self.base_preset.model_version, 'light-test')
        self.assertEqual(self.base_preset.base_data_directory, '/path/to/data')
        self.assertEqual(
            self.base_preset.base_output_directory, '/path/to/output')
        self.assertIn('/path/to/data/lion', self.base_preset.lion_directories)
        self.assertIn('/path/to/data/no_lion',
                      self.base_preset.no_lion_directories)
        self.assertTrue(hasattr(self.base_preset, 'with_augmentation'))
        self.assertEqual(self.base_preset.batch_size, 2)
        self.assertEqual(self.base_preset.alpha, 1e-3)


class TestPreset01(unittest.TestCase):
    """
    Test preset 1.
    """

    def test_configuration(self):
        """
        Test configuration.
        """
        preset = Preset01()
        self.assertEqual(preset.notebook_number, 1)
        self.assertEqual(preset.epochs, 2400)
        self.assertEqual(preset.image_dimensions, (128, 128))
        self.assertFalse(preset.with_augmentation)
        self.assertEqual(preset.batch_size, 16)
        self.assertEqual(preset.model_version, "light")
        self.assertEqual(preset.color_mode, 'grayscale')
        self.assertAlmostEqual(preset.alpha, 1e-5)
        self.assertIn('lion', preset.lion_directories[0])
        self.assertIn('no_lion', preset.no_lion_directories[0])
