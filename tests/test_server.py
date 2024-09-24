"""
Test pumaguard-server
"""

import unittest
from unittest.mock import patch
import sys

from pumaguard.cmd import server


class TestServer(unittest.TestCase):
    """
    Test pumaguard-server.
    """

    def setUp(self):
        self.server = server.Server()
        self.server.app.testing = True
        self.client = self.server.app.test_client()

    def test_parse_commandline(self):
        """
        Test whether we can parse command line arguments.
        """
        with patch.object(sys, 'argv', ['command', '--help']):
            with patch('sys.exit') as mock_exit:
                _ = server.parse_commandline()
                mock_exit.assert_called()

    def test_parse_commandline_defaults(self):
        """
        Test default command line arguments.
        """
        with patch.object(sys, 'argv', ['command']):
            args = server.parse_commandline()
            self.assertEqual(args.host, '0.0.0.0')
            self.assertEqual(args.port, 1443)
            self.assertFalse(args.debug)

    def test_parse_commandline_custom_host_port(self):
        """
        Test custom host and port arguments.
        """
        with patch.object(sys, 'argv', ['command', '--host', '127.0.0.1',
                                        '--port', '8080']):
            args = server.parse_commandline()
            self.assertEqual(args.host, '127.0.0.1')
            self.assertEqual(args.port, 8080)

    def test_parse_commandline_debug_mode(self):
        """
        Test enabling debug mode.
        """
        with patch.object(sys, 'argv', ['command', '--debug']):
            args = server.parse_commandline()
            self.assertTrue(args.debug)

    def test_classify_image_no_image(self):
        """
        Test classify_image endpoint with no image provided.
        """
        response = self.client.post('/classify', json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json, {'error': 'No image provided'})

    def test_classify_image_with_image(self):
        """
        Test classify_image endpoint with an image provided.
        """
        response = self.client.post(
            '/classify', json={'image': 'fake_image_data'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json, {'classification': 'puma', 'confidence': 0.95})
