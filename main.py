"""PyInstaller entry point for launching the Corrosion Simulator Streamlit application."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import time
import webbrowser


def _configure_local_no_proxy() -> None:
    """Ensure corporate HTTP(S)_PROXY settings never intercept local Streamlit traffic."""
    required_hosts = ["localhost", "127.0.0.1", "::1"]
    for env_key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_key, "")
        tokens = [item.strip() for item in existing.split(",") if item.strip()]
        for host in required_hosts:
            if host not in tokens:
                tokens.append(host)
        os.environ[env_key] = ",".join(tokens)


def _ensure_standard_streams() -> None:
    """Guard against None stdout/stderr if packaged with --noconsole on Windows."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _find_available_port(host: str = "127.0.0.1", preferred_port: int = 8501, max_tries: int = 10) -> int:
    """Return ``preferred_port`` if free, else scan sequential fallback ports."""
    for offset in range(max_tries):
        candidate = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return preferred_port


def _wait_and_open_browser(host: str, port: int, timeout_s: float = 60.0) -> None:
    """Poll the local Streamlit TCP socket until listening, then open the default browser."""
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}"
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.4)
    # Fallback attempt if startup took longer than timeout_s
    webbrowser.open(url)


def main() -> None:
    _ensure_standard_streams()
    _configure_local_no_proxy()

    # Import package helpers & hint dynamic backends for PyInstaller's import graph
    import matplotlib.backends.backend_agg  # noqa: F401
    import matplotlib.backends.backend_pdf  # noqa: F401
    import matplotlib.backends.backend_svg  # noqa: F401
    import plotly  # noqa: F401
    import pyarrow.parquet  # noqa: F401
    import streamlit.web.cli as stcli

    from corrosion_simulator.paths import (
        DATA_DIR_ENV_VAR,
        get_bundle_dir,
        get_external_data_dir,
        is_frozen,
    )

    bundle_dir = get_bundle_dir()
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))

    external_data_dir: Path = get_external_data_dir()
    os.environ[DATA_DIR_ENV_VAR] = str(external_data_dir)

    # Ensure external data/raw directory exists next to the exe so users can drop CSVs easily
    if is_frozen():
        try:
            (external_data_dir / "raw").mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    app_path = bundle_dir / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Cannot locate bundled Streamlit entry script: {app_path}")

    host = "127.0.0.1"
    port = _find_available_port(host=host, preferred_port=8501)

    print(f"[CorrosionSimulator] App script : {app_path}")
    print(f"[CorrosionSimulator] Data folder: {external_data_dir}")
    print(f"[CorrosionSimulator] Server URL : http://{host}:{port}")

    threading.Thread(
        target=_wait_and_open_browser,
        args=(host, port),
        daemon=True,
    ).start()

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={port}",
        f"--server.address={host}",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ]
    stcli.main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
