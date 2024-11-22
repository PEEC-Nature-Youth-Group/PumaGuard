"""
Test trailcam
"""

import unittest
from unittest.mock import patch
import sys
import requests

from pumaguard import trailcam


class TestTrailcam(unittest.TestCase):
    """
    Test pumaguard-server.
    """

    def test_parse_commandline(self):
        """
        Test whether we can parse command line arguments.
        """
        with patch.object(sys, 'argv', ['command', '--help']):
            with patch('sys.exit') as mock_exit:
                _ = trailcam.parse_commandline()
                mock_exit.assert_called()

    @patch('pumaguard.trailcam.requests.post')
    def test_send_image_to_server_success(self, mock_post):
        """
        Test successful image upload to server.
        """
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'status': 'success'}
        response = trailcam.send_image_to_server(
            'data/lion/PICT0001.JPG',
            'http://example.com/upload',
        )
        self.assertEqual(response, {'status': 'success'})
        mock_post.assert_called_once()

    @patch('pumaguard.trailcam.requests.post')
    def test_send_image_to_server_failure(self, mock_post):
        """
        Test failed image upload to server.
        """
        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = \
            requests.exceptions.HTTPError
        with self.assertRaises(requests.exceptions.HTTPError):
            trailcam.send_image_to_server(
                'data/lion/PICT0001.JPG',
                'http://example.com/upload',
            )
        mock_post.assert_called_once()
