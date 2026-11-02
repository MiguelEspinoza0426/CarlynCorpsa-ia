"""Video sources: cameras (RTSP/webcam/file) and, on Windows, a desktop window (iVMS-4200)."""
import os
from urllib.parse import quote, urlsplit, urlunsplit

from .camera import CameraReader


class _WindowSource:
    """Adapts the window reader (LectorVentana) to the read()/close() interface."""

    def __init__(self, spec: str):
        from .window import LectorVentana
        self._reader = LectorVentana(spec)
        self.connected, self.error = True, None

    def read(self):
        return self._reader.leer()

    def close(self):
        self._reader.cerrar()


def with_credentials(url: str, credentials_ref: str | None) -> str:
    """Adds user/password from <REF>_USER / <REF>_PASS environment variables to an RTSP URL.

    Credentials never travel through the API: each inference node keeps them in its environment.
    """
    if not credentials_ref:
        return url
    user = os.environ.get(f"{credentials_ref}_USER", "")
    password = os.environ.get(f"{credentials_ref}_PASS", "")
    if not user:
        return url
    parts = urlsplit(url)
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def open_source(source: str, name: str = ""):
    """'0' -> webcam, 'window:<title>' -> desktop window, anything else (URL or file) -> CameraReader."""
    if source.startswith("window:"):
        return _WindowSource(source[len("window:"):])
    return CameraReader(int(source) if source.isdigit() else source, name)
