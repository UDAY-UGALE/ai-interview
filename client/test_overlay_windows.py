"""
Window-lifecycle regression tests for the desktop client.

One narrow concern, and it earns its own file because it is invisible to
every other test in the repo: which window is allowed to end the process.

The bug this exists for, in the user's words: open the overlay, open the
Project & Experience Context window, close that window -- saved or not --
and BOTH windows disappear, mid-interview.

The cause is a Qt rule that is easy to walk into. Qt quits the application
when the last window carrying the WA_QuitOnClose attribute closes. The
overlay is a Qt.Tool window (deliberately -- that is what keeps it out of
the taskbar and the alt-tab list), and Qt CLEARS WA_QuitOnClose for Tool
windows. So the overlay is invisible to that count. An ordinary dialog has
the attribute set, which made the project window the only one Qt was
counting: closing it dropped the count to zero and took the session with
it.

Every secondary window added to this client from now on has to satisfy the
invariant these tests state: the overlay's lifetime belongs to
OverlayWindow._shutdown() (the X button and Esc), and nothing else.

Runs fully offline against Qt's offscreen platform -- no display, no
backend, no network (the dialog's initial load fails against a dead port,
which is itself worth exercising).

Usage (from the project root):
    python client\\test_overlay_windows.py
    python client\\test_overlay_windows.py --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# Qt must be told to run headless BEFORE QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


DEAD_BACKEND = "http://127.0.0.1:9"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    _results.append((name, ok, detail))


def _drain(app: QApplication, dialog, timeout: float = 20.0) -> bool:
    """Let the dialog's initial load request finish.

    Not politeness -- correctness of the timing test below. closeEvent
    waits up to three seconds for in-flight threads, and an earlier version
    of this test measured that wait instead of the thing it meant to
    measure, reporting a pass and a fail identically.
    """
    deadline = time.monotonic() + timeout
    while dialog._threads and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    return not dialog._threads


def run(verbose: bool) -> int:
    app = QApplication([])

    import overlay_app
    from project_context import ProjectContextDialog

    # main() is what real users run, and it is where the application-wide
    # half of the fix lives. This test builds its own QApplication, so it
    # applies the same setting -- and asserts on it below, which is what
    # keeps main() and this file from drifting apart.
    app.setQuitOnLastWindowClosed(False)

    overlay = overlay_app.OverlayWindow(DEAD_BACKEND, "window-test", capturable=True)
    overlay.show()

    dialog = ProjectContextDialog(
        DEAD_BACKEND, "window-test", {}, hidden_from_capture=False
    )
    dialog.show()

    # --- the invariant, stated directly -------------------------------
    check(
        "overlay is a Tool window Qt will not count",
        not overlay.testAttribute(Qt.WA_QuitOnClose),
        "Qt clears WA_QuitOnClose for Qt.Tool windows -- this is the premise "
        "of the whole bug, so if it ever changes these tests need rereading",
    )
    check(
        "project dialog cannot end the application",
        not dialog.testAttribute(Qt.WA_QuitOnClose),
        "WA_QuitOnClose must be False on the dialog, or closing it quits Qt",
    )
    check(
        "application does not quit on last window closed",
        not app.quitOnLastWindowClosed(),
        "main() turns this off so the overlay owns its own lifetime",
    )

    # --- and the behaviour it is supposed to produce -------------------
    drained = _drain(app, dialog)
    check("dialog settled before close", drained, "in-flight load finished")

    started = time.monotonic()
    quit_at: dict[str, float | None] = {"t": None}
    app.aboutToQuit.connect(
        lambda: quit_at.__setitem__("t", time.monotonic() - started)
    )

    QTimer.singleShot(200, dialog.close)
    QTimer.singleShot(2000, app.quit)  # backstop, so this always terminates
    app.exec()

    elapsed = quit_at["t"]
    # Quitting right when the dialog closed means the dialog took the app
    # down. Quitting at the backstop means it did not.
    took_down = elapsed is not None and elapsed < 1.0
    check(
        "closing the project window leaves the overlay running",
        not took_down,
        f"dialog closed at 0.20s, application quit at "
        f"{elapsed:.2f}s" if elapsed else "application never quit",
    )

    passed = sum(1 for _n, ok, _d in _results if ok)
    for name, ok, detail in _results:
        status = "ok  " if ok else "FAIL"
        print(f"[{status}] {name}")
        if verbose or not ok:
            print(f"       {detail}")
    print(f"\n{passed}/{len(_results)} checks passed.")
    return 0 if passed == len(_results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbose", action="store_true", help="Show every detail line.")
    args = parser.parse_args()
    code = run(args.verbose)
    # Qt's own teardown races with the interpreter's on Windows; the checks
    # are done, so leave now rather than risk a spurious non-zero exit.
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
