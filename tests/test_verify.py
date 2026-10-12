"""
Unit tests for accuracy and loss functions in pumaguard.verify.
"""

import argparse
import os
import tempfile
import unittest
from unittest.mock import (
    MagicMock,
    call,
    patch,
)

from pumaguard.verify import (
    configure_subparser,
    get_accuracy,
    get_binary_accuracy,
    get_crossentropy_loss,
    get_mean_squared_error,
    main,
    verify_model,
)


class TestVerify(unittest.TestCase):
    """
    Test verify action.
    """

    def setUp(self):
        self.predictions = [
            (
                "lion/SYFW0033.JPG",
                0.0017,
                0,
            ),
            (
                "lion/SYFW0271.JPG",
                0.0436,
                0,
            ),
            (
                "lion/SYFW0278.JPG",
                0.0212,
                0,
            ),
            (
                "lion/SYFW1932.JPG",
                0.5652,
                0,
            ),
            (
                "lion/SYFW5844.JPG",
                0.8118,
                0,
            ),
            (
                "no-lion/SYFW0022.JPG",
                0.3615,
                1,
            ),
            (
                "no-lion/SYFW0084.JPG",
                0.8778,
                1,
            ),
            (
                "no-lion/SYFW0119.JPG",
                0.3408,
                1,
            ),
            (
                "no-lion/SYFW0186.JPG",
                0.7188,
                1,
            ),
            (
                "no-lion/SYFW0197.JPG",
                0.9999,
                1,
            ),
            (
                "no-lion/SYFW0220.JPG",
                0.9987,
                1,
            ),
            (
                "no-lion/SYFW0235.JPG",
                0.2738,
                1,
            ),
            (
                "no-lion/SYFW0761.JPG",
                0.8309,
                1,
            ),
            (
                "no-lion/SYFW1915.JPG",
                0.4154,
                1,
            ),
            (
                "no-lion/SYFW6331.JPG",
                0.6892,
                1,
            ),
            (
                "no-lion/SYFW6550.JPG",
                0.8819,
                1,
            ),
        ]

    def test_get_accuracy(self):
        """
        Tests the accuracy calculation for given predictions.
        """
        self.assertAlmostEqual(
            get_accuracy(self.predictions), 0.6840755, places=6
        )

    def test_get_binary_accuracy(self):
        """
        Tests the binary accuracy calculation for given predictions.
        """
        self.assertAlmostEqual(
            get_binary_accuracy(self.predictions), 0.625, places=6
        )

    def test_get_crossentropy_loss(self):
        """
        Tests the cross-entropy loss calculation for given predictions.
        """
        loss = get_crossentropy_loss(self.predictions)
        self.assertAlmostEqual(loss, 0.4989816722098235, places=6)

    def test_get_mean_squared_error(self):
        """
        Tests the mean squared error calculation for given predictions.
        """
        mse = get_mean_squared_error(self.predictions)
        self.assertAlmostEqual(mse, 0.18283256874999995, places=6)

    def test_configure_subparser_accepts_path_and_images(self):
        """The verify command accepts an optional data path and image list."""
        parser = argparse.ArgumentParser()
        configure_subparser(parser)

        options = parser.parse_args(
            ["--verification-path", "/data", "first.jpg", "second.jpg"]
        )

        self.assertEqual(options.verification_path, "/data")
        self.assertEqual(options.image, ["first.jpg", "second.jpg"])

    @patch("pumaguard.verify.classify_image")
    def test_verify_model_reports_sorted_predictions_and_metrics(
        self, mock_classify
    ):
        """Verification labels files correctly and reports metrics."""
        mock_classify.side_effect = [0.2, 0.8, 0.8, 0.2]
        presets = MagicMock()

        with tempfile.TemporaryDirectory() as verification_path:
            lion_dir = os.path.join(verification_path, "lion")
            no_lion_dir = os.path.join(verification_path, "no-lion")
            os.mkdir(lion_dir)
            os.mkdir(no_lion_dir)
            for filename in ("zebra.jpg", "antelope.jpg"):
                filepath = os.path.join(lion_dir, filename)
                with open(filepath, "w", encoding="utf-8"):
                    pass
            for filename in ("yak.jpg", "bear.jpg"):
                filepath = os.path.join(no_lion_dir, filename)
                with open(filepath, "w", encoding="utf-8"):
                    pass
            presets.verification_path = verification_path

            with patch("builtins.print") as mock_print:
                verify_model(presets)

        expected_paths = [
            os.path.relpath(os.path.join(lion_dir, "antelope.jpg"), "."),
            os.path.relpath(os.path.join(lion_dir, "zebra.jpg"), "."),
            os.path.relpath(os.path.join(no_lion_dir, "bear.jpg"), "."),
            os.path.relpath(os.path.join(no_lion_dir, "yak.jpg"), "."),
        ]
        self.assertEqual(
            mock_classify.call_args_list,
            [call(presets, path) for path in expected_paths],
        )
        self.assertEqual(
            mock_print.call_args_list,
            [
                call("number false positives = 1"),
                call("number false negatives = 1"),
                call("out of 2 lion and 2 no-lion images"),
                call("accuracy           = 50.00%"),
                call("binary accuracy    = 50.00%"),
                call("crossentropy loss  = 0.9163"),
                call("mean squared error = 0.3400"),
            ],
        )

    @patch("pumaguard.verify.logger")
    def test_get_accuracy_warns_for_negative_predictions(self, mock_logger):
        """Invalid predictions are logged for either expected label."""
        predictions = [("lion.jpg", -1.0, 0), ("other.jpg", -1.0, 1)]

        self.assertEqual(get_accuracy(predictions), 0)

        self.assertEqual(mock_logger.warning.call_count, 2)

    @patch("pumaguard.verify.verify_model")
    def test_main_delegates_to_verify_model(self, mock_verify_model):
        """The command entry point uses the supplied settings."""
        presets = MagicMock()

        main(argparse.Namespace(), presets)

        mock_verify_model.assert_called_once_with(presets)
