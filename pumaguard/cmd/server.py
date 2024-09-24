"""
PumaGuard Server

This script will run the machine learning model and receive images from the
trailcams for classification. It will contact the speakers and/or lights if a
puma was identified.
"""

import argparse
from flask import (
    Flask,
    request,
    jsonify,
)


class Server:
    """
    The Pumaguard-server.
    """

    def __init__(self) -> None:
        """
        Create the app.
        """
        self.app = Flask("PumaGuard")
        self.register_routes()

    def register_routes(self):
        """
        Register all routes.
        """
        self.app.add_url_rule('/classify', 'classify_image',
                              self.classify_image, methods=['POST'])

    def classify_image(self):
        """
        Endpoint to classify an image.
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
    Entry point.
    """
    options = parse_commandline()
    server = Server()
    server.app.run(host=options.host, port=options.port, debug=options.debug)
