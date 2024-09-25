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

        Returns:
            Response: JSON response.
        """
        print('starting')
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
            return jsonify({'error': f'Illegal data provided: {e}'}), 400
        _ = self.classify_image(image)
        return jsonify({}), 200

    def classify_image(self, image) -> float:
        """
        Classify the image.

        Args:
            image (Image): The image.

        Returns:
            float: The probability the image does not contain a Puma, i.e. a
            value of 0 corresponds to 100% Puma, while a value of 1 corresponds
            to 100% no Puma.
        """
        print(f'received {type(image)}')
        return 0.8


def parse_commandline() -> argparse.Namespace:
    """
    Parse command line arguments.
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
    options = parse_commandline()
    server = Server()
    server.app.run(host=options.host, port=options.port, debug=options.debug)
