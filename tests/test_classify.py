"""
Tests for classify script.
"""

import unittest
from unittest.mock import patch, MagicMock

import os
import sys
import numpy as np

from pumaguard.classify import (
    parse_commandline,
    split_argument,
    classify_images,
    main,
)


class TestParseCommandline(unittest.TestCase):
    """
    Unit tests for parse_commandline function
    """

    @patch.object(sys, 'argv', ['pumaguard-classify', '--notebook', '2',
                                'image1.jpg', 'image2.jpg:no-lion'])
    def test_parse_commandline_with_all_arguments(self):
        """
        Test with all arguments.
        """
        options = parse_commandline()
        self.assertEqual(options.notebook, 2)
        self.assertEqual(options.image, ['image1.jpg', 'image2.jpg:no-lion'])

    @patch.object(sys, 'argv', ['pumaguard-classify', 'image1.jpg'])
    def test_parse_commandline_with_minimal_arguments(self):
        """
        Test with minimal arguments.
        """
        options = parse_commandline()
        self.assertEqual(options.notebook, 1)
        self.assertEqual(options.image, ['image1.jpg'])

    @patch.object(sys, 'argv', ['pumaguard-classify', '--completion', 'bash'])
    def test_parse_commandline_with_completion(self):
        """
        Test completions.
        """
        with self.assertRaises(SystemExit):
            parse_commandline()

    @patch.object(sys, 'argv', ['pumaguard-classify'])
    def test_parse_commandline_missing_image(self):
        """
        Test missing image.
        """
        with self.assertRaises(ValueError):
            parse_commandline()


class TestSplitArgument(unittest.TestCase):
    """
    Unit tests for split_argument function
    """

    def test_split_argument_with_label(self):
        """
        Test splitting argument with label.
        """
        filename, label = split_argument('image1.jpg:lion')
        self.assertEqual(filename, 'image1.jpg')
        self.assertEqual(label, 'lion')

    def test_split_argument_without_label(self):
        """
        Test splitting argument without label.
        """
        filename, label = split_argument('image1.jpg')
        self.assertEqual(filename, 'image1.jpg')
        self.assertEqual(label, 'lion')  # Verify if default label is 'lion'

    def test_split_argument_invalid_label(self):
        """
        Test splitting argument with invalid label.
        """
        with self.assertRaises(ValueError):
            split_argument('image1.jpg:invalid')


class TestClassifyImages(unittest.TestCase):
    """
    Unit tests for classify_images function
    """

    @patch('pumaguard.classify.keras.preprocessing.image_dataset_from_directory')  # pylint: disable=line-too-long
    @patch('pumaguard.classify.keras.Model.predict')
    def test_classify_images(self, mock_predict,
                             mock_image_dataset_from_directory):
        """
        Test classify images.
        """
        mock_predict.return_value = np.array([[0.2]])
        probabilities = MagicMock()
        probabilities.return_value = 0
        mock_image_dataset_from_directory.return_value = [([probabilities], [MagicMock()])]

        presets = MagicMock()
        presets.model_version = 'pre-trained'
        presets.batch_size = 1
        presets.image_dimensions = (224, 224)
        presets.notebook = 1

        model = MagicMock()
        # model.predict = MagicMock()
        # model.predict.return_value = [0.2]

        classify_images(presets, model, 'test_image_path')

        mock_image_dataset_from_directory.assert_called_once_with(
            'test_image_path',
            batch_size=1,
            image_size=(224, 224),
            shuffle=True,
            color_mode='rgb',
        )
        mock_predict.assert_called_once()


class TestMain(unittest.TestCase):
    """
    Unit tests for main function
    """

    @patch('pumaguard.classify.parse_commandline')
    @patch('pumaguard.classify.split_argument')
    @patch('pumaguard.classify.shutil.copy')
    @patch('pumaguard.classify.tempfile.TemporaryDirectory')
    @patch('pumaguard.classify.classify_images')
    def test_main(self, mock_classify_images, MockTemporaryDirectory,  # pylint: disable=invalid-name
                  mock_copy, mock_split_argument, mock_parse_commandline):
        """
        Test main function.
        """
        mock_parse_commandline.return_value = MagicMock(
            image=['image1.jpg:lion'])
        mock_split_argument.return_value = ('image1.jpg', 'lion')
        mock_temp_dir = MagicMock()
        MockTemporaryDirectory.return_value.__enter__.return_value = \
            mock_temp_dir

        main()

        mock_parse_commandline.assert_called_once()
        mock_split_argument.assert_called_once_with('image1.jpg:lion')
        mock_copy.assert_called_once_with(
            'image1.jpg', os.path.join(mock_temp_dir, 'lion'))
        mock_classify_images.assert_called_once()
