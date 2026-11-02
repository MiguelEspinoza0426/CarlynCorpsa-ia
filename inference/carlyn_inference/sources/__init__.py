"""Video sources of a camera, chosen from its stream URL (see docs/contrato-integracion.md):

- rtsp://, rtsps://, http://, https:// -> IP camera or DVR channel (credentials from the node environment);
- webcam:<n>                            -> camera <n> connected to the inference PC (0 = first one);
- window:<title>[:x,y,w,h]              -> a window of the inference PC desktop, e.g. Hik-Connect or iVMS-4200;
- anything else (digits or a file path) -> local overrides for tests (SOURCE_OVERRIDES).
"""
import os
import re
from urllib.parse import quote, urlsplit, urlunsplit

from .camera import CameraReader

NETWORK_SCHEMES = ("rtsp", "rtsps", "http", "https")
WEBCAM = re.compile(r"webcam:(\d{1,2})")
WINDOW_PREFIX = "window:"


def with_credentials(url: str, credentials_ref: str | None) -> str:
    """Adds user/password from <REF>_USER / <REF>_PASS environment variables to an RTSP URL.

    Credentials never travel through the API: each inference node keeps them in its environment.
    Local sources (webcam, window) are returned unchanged.
    """
    if not credentials_ref or urlsplit(url).scheme not in NETWORK_SCHEMES:
        return url
    user = os.environ.get(f"{credentials_ref}_USER", "")
    password = os.environ.get(f"{credentials_ref}_PASS", "")
    if not user:
        return url
    parts = urlsplit(url)
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def open_source(source: str, name: str = ""):
    """Opens the reader for a source: webcam index, desktop window, or URL/file."""
    if source.startswith(WINDOW_PREFIX):
        from .window import WindowReader
        return WindowReader(source[len(WINDOW_PREFIX):])
    webcam = WEBCAM.fullmatch(source)
    if webcam or source.isdigit():
        return CameraReader(int(webcam.group(1) if webcam else source), name)
    return CameraReader(source, name)
