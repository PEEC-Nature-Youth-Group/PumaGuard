"""
Test pumaguard-server
"""

import unittest
from unittest.mock import patch
import sys
import base64

from pumaguard import server


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

    def test_classify_image_route_no_image(self):
        """
        Test classify_image endpoint with no image provided.
        """
        with patch('pumaguard.server.logger') as mock_logger:
            response = self.client.post('/classify', json={})
            self.assertEqual(response.status_code, 400)
            self.assertIn('Illegal data provided', response.json['error'])
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            print(call_args)
            print(call_args[0])
            self.assertIn('Illegal data provided', call_args[0][0])
            self.assertIn('No image provided',
                          call_args[0][1].get_description())

    def test_classify_image_route_with_image(self):
        """
        Test classify_image endpoint with an image provided.
        """
        with open('data/lion/PICT0001.JPG', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        response = self.client.post(
            '/classify', json={'image': encoded_image})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {})

    def test_classify_image_route_invalid_json(self):
        """
        Test classify_image endpoint with invalid JSON.
        """
        with patch('pumaguard.server.logger') as mock_logger:
            response = self.client.post(
                '/classify', data='invalid_json',
                content_type='application/json')
            self.assertEqual(response.status_code, 400)
            self.assertIn('Illegal data provided', response.json['error'])
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            self.assertIn('Illegal data provided', call_args[0][0])
            self.assertIn('The browser (or proxy) sent a request that '
                          'this server could not understand.',
                          call_args[0][1].get_description())

    def test_classify_image_invalid_image_data(self):
        """
        Test classify_image endpoint with invalid image data.
        """
        with patch('pumaguard.server.logger') as mock_logger:
            response = self.client.post(
                '/classify', json={'image': 'invalid_image_data'})
            self.assertEqual(response.status_code, 400)
            self.assertIn('Illegal data provided', response.json['error'])
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            self.assertIn('Illegal data provided', call_args[0][0])
            self.assertIn('Could not decode image',
                          call_args[0][1].get_description())
