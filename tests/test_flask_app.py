"""
Unit tests for the Flask web application.
"""

import unittest
from app import app


class TestFlaskApp(unittest.TestCase):
    def test_get_index(self):
        environ = {
            "PATH_INFO": "/",
            "REQUEST_METHOD": "GET",
            "wsgi.input": None
        }
        status_received = []
        headers_received = []

        def start_response(status, headers):
            status_received.append(status)
            headers_received.append(headers)

        body = app.wsgi_app(environ, start_response)
        self.assertIn("200 OK", status_received[0])
        self.assertGreater(len(body[0]), 0)
        self.assertIn(b"UPI Fraud Detection", body[0])

    def test_health_endpoint(self):
        environ = {
            "PATH_INFO": "/api/health",
            "REQUEST_METHOD": "GET",
            "wsgi.input": None
        }
        status_received = []

        def start_response(status, headers):
            status_received.append(status)

        body = app.wsgi_app(environ, start_response)
        self.assertIn("200 OK", status_received[0])
        self.assertIn(b"healthy", body[0])


if __name__ == '__main__':
    unittest.main()
