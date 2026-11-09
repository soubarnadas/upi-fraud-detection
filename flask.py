"""
Lightweight, standard-compliant micro-Flask implementation.
Provides Flask, request, render_template, and jsonify using Python's standard library
and Jinja2 for offline zero-dependency environments, while automatically delegating
to native Flask if installed.
"""

import sys
import os
import json
import urllib.parse
from wsgiref.simple_server import make_server
import jinja2

# Check if native flask is available in site-packages
_native_flask = None
try:
    import importlib.util
    # Find native flask avoiding self-recursion
    spec = importlib.util.find_spec("flask")
    if spec and spec.origin and not spec.origin.startswith(os.path.dirname(__file__)):
        _native_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_native_mod)
        _native_flask = _native_mod
except Exception:
    _native_flask = None


class RequestProxy:
    def __init__(self):
        self.method = "GET"
        self.form = {}
        self.args = {}
        self.json = None
        self.environ = {}

    def is_json(self):
        return self.json is not None


request = RequestProxy()


class Response:
    def __init__(self, body, status=200, content_type="text/html; charset=utf-8"):
        if isinstance(body, str):
            self.body = body.encode("utf-8")
        elif isinstance(body, bytes):
            self.body = body
        else:
            self.body = json.dumps(body).encode("utf-8")
        self.status = f"{status} OK" if status == 200 else f"{status} Response"
        self.content_type = content_type

    def __call__(self, environ, start_response):
        start_response(self.status, [("Content-Type", self.content_type)])
        return [self.body]


def jsonify(data):
    return Response(json.dumps(data), content_type="application/json")


def render_template(template_name, **context):
    template_dir = getattr(Flask._current_app, "template_folder", "templates")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    tpl = env.get_template(template_name)
    return Response(tpl.render(**context))


class Flask:
    _current_app = None

    def __init__(self, import_name, template_folder="templates", static_folder="static"):
        self.import_name = import_name
        self.template_folder = os.path.abspath(template_folder)
        self.static_folder = os.path.abspath(static_folder)
        self.routes = {}  # (rule, method) -> handler
        Flask._current_app = self

    def route(self, rule, methods=None):
        if methods is None:
            methods = ["GET"]

        def decorator(func):
            for method in methods:
                self.routes[(rule, method.upper())] = func
            return func

        return decorator

    def get(self, rule):
        return self.route(rule, methods=["GET"])

    def post(self, rule):
        return self.route(rule, methods=["POST"])

    def wsgi_app(self, environ, start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        # Update global request proxy
        global request
        request.method = method
        request.environ = environ

        # Parse query string
        query_string = environ.get("QUERY_STRING", "")
        request.args = urllib.parse.parse_qs(query_string)

        # Parse form / body data
        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0))
        except (ValueError, TypeError):
            content_length = 0

        request.form = {}
        request.json = None

        if content_length > 0:
            body = environ["wsgi.input"].read(content_length)
            content_type = environ.get("CONTENT_TYPE", "")

            if "application/json" in content_type:
                try:
                    request.json = json.loads(body.decode("utf-8"))
                except Exception:
                    request.json = None
            elif "application/x-www-form-urlencoded" in content_type:
                decoded = urllib.parse.parse_qs(body.decode("utf-8"))
                request.form = {k: v[0] if len(v) == 1 else v for k, v in decoded.items()}

        handler = self.routes.get((path, method))
        if not handler:
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"404 Not Found"]

        try:
            result = handler()
            if isinstance(result, Response):
                return result(environ, start_response)
            elif isinstance(result, str):
                resp = Response(result)
                return resp(environ, start_response)
            elif isinstance(result, (dict, list)):
                resp = jsonify(result)
                return resp(environ, start_response)
            elif isinstance(result, tuple):
                body, code = result[0], result[1]
                resp = Response(body, status=code)
                return resp(environ, start_response)
            return result
        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
            return [f"500 Internal Server Error: {str(e)}".encode("utf-8")]

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)

    def run(self, host="127.0.0.1", port=5000, debug=False):
        print(f" * Serving UPI Fraud Detection Flask app on http://{host}:{port}/")
        server = make_server(host, port, self.wsgi_app)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


# If native flask exists, expose it; otherwise use micro-framework
if _native_flask:
    Flask = _native_flask.Flask
    request = _native_flask.request
    jsonify = _native_flask.jsonify
    render_template = _native_flask.render_template
