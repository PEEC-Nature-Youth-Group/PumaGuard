"""
Test basic TensorFlow functions.
"""

import unittest
import tensorflow as tf


class TestTensorFlow(unittest.TestCase):
    """
    Test TensorFlow.
    """

    def test_tensorflow_version(self):
        """
        Check version.
        """
        expected_version = "2.17.1"
        self.assertEqual(tf.__version__, expected_version,
                         f"Expected TensorFlow version {expected_version}, " +
                         f"but got {tf.__version__}")

    def test_tensorflow_devices(self):
        """
        Check TensorFlow devices.
        """
        conf = tf.config.list_physical_devices('CPU')
        self.assertTrue(len(conf) > 0)
