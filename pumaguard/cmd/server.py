"""
PumaGuard Server

This script receive images from the Trailcam Units for classification by
running the machine learning model. If it is reasonably confident that a Puma
was detected  it will contact the relevant output units to activate the
speakers and lights.
"""

import argparse
import base64
import binascii
import io
import logging
from typing import Tuple
from flask import (
    Flask,
    jsonify,
    request,
    Response,
)
from werkzeug.exceptions import BadRequest
from PIL import (
    Image,
    UnidentifiedImageError,
)
from .. import __VERSION__

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Server:
    """
    The Pumaguard Server class.

    This class contains all of the routes to the server.
    """

    def __init__(self) -> None:
        """
        Create the app and register routes.
        """
        self.app = Flask("PumaGuard")
        self.register_routes()

    def register_routes(self) -> None:
        """
        Register all routes.
        """
        self.app.add_url_rule(
            '/classify',
            'classify_image',
            self.classify_image_route,
            methods=['POST'],
        )

    def classify_image_route(self) -> Tuple[Response, int]:
        """
        Endpoint to classify an image.

        The incoming request has the following fields:

        image: The base64 encoded image
        id: The trailcam ID (used to pair with proper Output Unit)

        Returns:
            Response: a JSON formatted response and a response code
        """
        try:
            data = request.json
            if data is None:
                raise BadRequest('no data provided')
            image_data = data.get('image')
            if not image_data:
                raise BadRequest('No image provided')
            try:
                image_bytes = base64.b64decode(image_data)
            except binascii.Error as e:
                raise BadRequest(f'Invalid base64 encoding: {e}') from e
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except UnidentifiedImageError as e:
                raise BadRequest(f'Could not decode image: {e}') from e
        except BadRequest as e:
            logger.error('Illegal data provided: %s', e)
            return jsonify({'error': 'Illegal data provided'}), 400
        # TODO The image should be placed in a queue for performance
        _ = self.classify_image(image)
        return jsonify({}), 200

    def classify_image(self, image: Image.Image) -> float:
        """
        Classify the image.

        Args:
            image (Image): The image.

        Returns:
            float: The probability the image does not contain a Puma, i.e. a
            value of 0 corresponds to 100% Puma, while a value of 1 corresponds
            to 100% no Puma.
        """
        # TODO Call the machine learning model
        # TODO If Puma was detected, contact the Output Unit
        print(f'received {type(image)}')
        return 0.8


def parse_commandline() -> argparse.Namespace:
    """
    Parse command line arguments.

    .. code-block::

        usage: pumaguard-server [-h] [--host HOST] [--port PORT] [--debug]

        options:
          -h, --help   show this help message and exit
          --host HOST  Host to listen on
          --port PORT  Port to listen on
          --debug      Enable debug mode
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host',
        help='Host to listen on',
        type=str,
        default='0.0.0.0',
    )
    parser.add_argument(
        '--port',
        help='Port to listen on',
        type=int,
        default=1443,
    )
    parser.add_argument(
        '--debug',
        help='Enable debug mode',
        action='store_true',
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the server.
    """
    print(f'Starting pumaguard-server v{__VERSION__}')
    options = parse_commandline()
    server = Server()
    server.app.run(host=options.host, port=options.port, debug=options.debug)
