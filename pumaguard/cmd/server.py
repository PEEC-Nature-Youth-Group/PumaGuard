"""
PumaGuard Server

This script receive images from the Trailcam Units for classification by
running the machine learning model. If it is reasonably confident that a Puma
was detected  it will contact the relevant output units to activate the
speakers and lights.
"""

import argparse
from flask import (
    Flask,
    jsonify,
    request,
    Response,
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
            self.classify_image,
            methods=['POST'],
        )

    def classify_image(self) -> Response:
        """
        Endpoint to classify an image.

        Returns:
            Response: JSON response containing the classification result or an
            error message.
        """
        data = request.json
        image = data.get('image')
        if not image:
            return jsonify({'error': 'No image provided'}), 400
        result = {'classification': 'puma', 'confidence': 0.95}
        return jsonify(result)


def parse_commandline() -> argparse.Namespace:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--port', type=int, default=1443,
                        help='Port to listen on')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the server.
    """
    options = parse_commandline()
    server = Server()
    server.app.run(host=options.host, port=options.port, debug=options.debug)
