from __future__ import annotations

import argparse
import contextlib
import functools
import socket
import sys
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path


LOCAL_BROWSER_HOSTS = {"127.0.0.1", "localhost"}


def _studio_dist_dir() -> Path:
    dist_dir = Path(__file__).resolve().parent / "studio" / "dist"
    if not dist_dir.exists():
        raise RuntimeError(
            "Studio frontend assets were not found in the installed package. "
            "Reinstall foreblocks with the studio extra or build the Studio assets first."
        )
    return dist_dir


def _is_port_available(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _resolve_port(host: str, requested_port: int) -> int:
    if _is_port_available(host, requested_port):
        return requested_port

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class _StudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        requested = self.translate_path(self.path)
        if Path(requested).exists():
            return super().do_GET()

        # SPA fallback: serve index.html for application routes.
        self.path = "/index.html"
        return super().do_GET()

    def log_message(self, format: str, *args):
        sys.stdout.write("[foreblocks-studio] " + format % args + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foreblocks-studio",
        description="Serve the bundled foreBlocks Studio frontend.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=5176, help="Preferred TCP port.")
    open_group = parser.add_mutually_exclusive_group()
    open_group.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Always open the Studio URL in the default browser after the server starts.",
    )
    open_group.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open the browser automatically.",
    )
    parser.set_defaults(open_browser=None)
    return parser


def should_open_browser(host: str, open_browser: bool | None) -> bool:
    if open_browser is not None:
        return bool(open_browser)
    return host in LOCAL_BROWSER_HOSTS


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dist_dir = _studio_dist_dir()
    host = str(args.host)
    port = _resolve_port(host, int(args.port))
    handler = functools.partial(_StudioHandler, directory=str(dist_dir))
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    open_browser = should_open_browser(host, args.open_browser)

    print(f"foreBlocks Studio serving {dist_dir}")
    print(f"Open {url}")
    if open_browser:
        print("Browser launch policy: auto-open enabled")
    else:
        print("Browser launch policy: auto-open disabled")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down foreBlocks Studio server.")
    finally:
        server.server_close()

    return HTTPStatus.OK


if __name__ == "__main__":
    raise SystemExit(main())
