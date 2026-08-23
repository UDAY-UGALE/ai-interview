"""Where the client's server settings come from.

An installed .exe cannot rely on environment variables the way a developer
running `python client\\overlay_app.py` can -- double-clicking an icon gives
you whatever the shell had, which is nothing. But a developer must not have
to maintain a config file either, and neither of them should ever edit
source to change a URL.

So there is one precedence chain, and every entry point uses it:

    1. an explicit CLI flag        (--api-url / --token / ...)
    2. an environment variable     (INTERVIEW_API_URL, ...)
    3. a config file               (see _config_paths below)
    4. localhost:8000, no token    (a working local dev default)

Configuration is deployment, not behaviour: pointing the same build at a
laptop, at Render, and later at AWS is a config change in all three cases,
never a rebuild.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_API_URL = "http://127.0.0.1:8000"
CONFIG_FILENAME = "interviewcopilot.json"

# Set by load_config() so the file is read once per process rather than once
# per lookup -- the overlay asks for settings from several places.
_file_values: dict | None = None


def _app_dir() -> Path:
    """The directory the user actually installed into.

    Frozen by PyInstaller, __file__ points inside a temporary extraction
    directory that is deleted on exit, so a config file next to it would be
    unreachable. sys.executable is the .exe the user launched.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _config_paths() -> list[Path]:
    paths = []

    explicit = os.environ.get("INTERVIEW_CONFIG")
    if explicit:
        paths.append(Path(explicit).expanduser())

    # Next to the executable: what an installer or a downloaded zip ships,
    # so one download works with no setup at all.
    paths.append(_app_dir() / CONFIG_FILENAME)
    paths.append(_app_dir().parent / CONFIG_FILENAME)

    # Per-user, survives reinstalling over the top.
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "InterviewCopilot" / CONFIG_FILENAME)
    paths.append(Path.home() / ".config" / "interviewcopilot" / CONFIG_FILENAME)

    return paths


def _load_file_values() -> dict:
    global _file_values
    if _file_values is not None:
        return _file_values

    _file_values = {}
    for path in _config_paths():
        try:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A broken config file must not stop the app starting -- fall
            # through to env vars and defaults, which still work.
            continue
        if isinstance(data, dict):
            _file_values = data
            break
    return _file_values


def config_value(key: str, env_var: str, default: str = "") -> str:
    """One setting, resolved from env then file then default.

    CLI flags are not handled here: argparse applies them by using the
    result of this call as its `default=`, so an explicitly passed flag
    naturally wins.
    """
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env

    value = _load_file_values().get(key)
    return str(value) if value else default


def http_to_ws(api_url: str) -> str:
    if api_url.startswith("https://"):
        return "wss://" + api_url[len("https://") :]
    if api_url.startswith("http://"):
        return "ws://" + api_url[len("http://") :]
    return api_url


@dataclass(frozen=True)
class ClientDefaults:
    api_url: str
    ws_url: str
    token: str
    session_id: str


def load_defaults() -> ClientDefaults:
    """Defaults for argparse. Call once at startup."""
    api_url = config_value("api_url", "INTERVIEW_API_URL", DEFAULT_API_URL).rstrip("/")

    # ws_url is derived from api_url unless it is set explicitly -- one URL
    # to configure in the normal case, an override for the unusual one (a
    # proxy that terminates websockets somewhere else).
    ws_url = config_value("ws_url", "INTERVIEW_WS_URL", "")
    if not ws_url:
        ws_url = f"{http_to_ws(api_url)}/ws/audio"

    return ClientDefaults(
        api_url=api_url,
        ws_url=ws_url,
        token=config_value("token", "INTERVIEW_TOKEN", ""),
        # Sessions are keyed by this string: two clients sharing one would
        # share a resume, a history, and each other's answers. The Windows
        # username is a reasonable per-machine default.
        session_id=config_value(
            "session_id", "INTERVIEW_SESSION_ID", os.environ.get("USERNAME") or "default"
        ),
    )


def config_file_locations() -> list[str]:
    """For --where-config / support questions: the paths actually searched."""
    return [str(p) for p in _config_paths()]
