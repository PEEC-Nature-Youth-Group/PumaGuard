"""
PumaGuard Trailcam Unit

This script will monitor the trailcam attached to it and send images to the
server unit.
"""

import argparse
import base64
import requests


def send_image_to_server(image_path: str, server_url: str):
    """
    Send an image to the server via a REST API.

    :param image_path: Path to the image file.
    :param server_url: URL of the server to send the image to.
    """
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        data = {'image': encoded_image}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            server_url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()


def parse_commandline() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="Send FILE to server",
        type=str,
    )
    parser.add_argument(
        "--host",
        help="Server host",
        type=str,
        default="0.0.0.0"
    )
    parser.add_argument(
        "--port",
        help="Server port",
        type=int,
        default=1443
    )
    return parser.parse_args()


def main():
    """
    Entry point.
    """
    options = parse_commandline()
    if options.file:
        server_url = f"http://{options.host}:{options.port}/classify"
        send_image_to_server(options.file, server_url)
