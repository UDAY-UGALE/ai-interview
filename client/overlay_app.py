"""
Floating overlay window that shows live InterviewCopilot answers on top of
Teams/Meet/Zoom, the way Parakeet AI / Chiku AI style overlays work.

- Frameless, always-on-top, semi-transparent panel you can drag anywhere,
  and resize from any edge/corner.
- Connects to /ws/answers?session_id=... and streams tokens live as they
  arrive from the backend, revealed at a readable typing pace instead of
  dumping the whole answer the instant it finishes generating.
- Lets you pick the answer LLM provider/model from a dropdown -- this is
  pushed to the backend session via POST /session (partial update, so it
  doesn't touch your resume/JD) and takes effect on the next question.
- Optionally excluded from screen capture (Windows only) via
  SetWindowDisplayAffinity, so screen-share/recording tools don't see it --
  this mirrors how Parakeet/Chiku-style overlays behave. Disable with
  --capturable if you don't want that.

Requires: PySide6, websockets  (pip install -r requirements.txt)

Usage:
    python client\\overlay_app.py --session-id default
    python client\\overlay_app.py --session-id default --api-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import ctypes
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request

import requests
import websockets
from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
    QPoint,
    QRect,
    QTimer,
    QObject,
    QEvent,
    QBuffer,
    QIODevice,
    QPropertyAnimation,
    QEasingCurve,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QFont,
    QKeySequence,
    QShortcut,
    QTextCursor,
    QTextBlockFormat,
    QTextCharFormat,
    QSyntaxHighlighter,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_API_URL = "http://127.0.0.1:8000"
RESIZE_MARGIN = 8
MIN_WIDTH = 280
MIN_HEIGHT = 180

# Roughly how fast revealed text should appear, in characters per second.
# Tuned to look like a person typing/thinking, fast enough to stay ahead of
# you reading it, slow enough not to dump the whole answer instantly.
REVEAL_CHARS_PER_SECOND = 75
REVEAL_TICK_MS = 15

# Windows SetWindowDisplayAffinity flag: hides the window from screen
# capture (screen share, screen recording, snipping tool) while it stays
# fully visible to you locally.
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WDA_NONE = 0x00000000

_QUESTION_STYLE_NORMAL = "color: #ffd479; font-size: 13px; font-weight: 600;"
_QUESTION_STYLE_LOW_CONFIDENCE = "color: #ff8a8a; font-size: 13px; font-weight: 600;"

# Activity states shown by the pulsing indicator: state -> (label text, color).
_ACTIVITY_STATES = {
    "listening": ("Listening", "#7CFC9E"),
    "hearing": ("Hearing you", "#9adbff"),
    "thinking": ("Thinking", "#ffd479"),
    "answering": ("Answering", "#ffa64d"),
}
ACTIVITY_PULSE_MS = 450


def _http_get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _http_post_json(url: str, payload: dict, timeout: float = 5.0) -> None:
    """Fire-and-forget POST, run off the UI thread by the caller."""
    try:
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(request, timeout=timeout)
    except (urllib.error.URLError, TimeoutError):
        pass


class _AnswerScrollNav(QObject):
    """Installed on the answer view's viewport: scrolling over the answer
    text browses Q&A history (up = older, down = newer/live) instead of
    scrolling within the (usually short, non-scrolling) text itself."""

    def __init__(self, window: "OverlayWindow") -> None:
        super().__init__(window)
        self._window = window

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QEvent.Wheel:
            if event.angleDelta().y() > 0:
                self._window._show_previous()
            else:
                self._window._show_next()
            return True
        return False


class _AnswerSyntaxHighlighter(QSyntaxHighlighter):
    """Lightweight regex-based syntax highlighter, scoped to only fenced
    code blocks -- setMarkdown() marks each code-block paragraph with
    QTextBlockFormat.NonBreakableLines (Qt's own signal for 'this came from
    a ``` fence'), so that's used to skip highlighting regular answer text
    entirely and only color actual code. Not a full tokenizer -- good
    enough to make Python/JS/etc snippets readable at a glance instead of
    flat white-on-dark text."""

    _KEYWORDS = re.compile(
        r"\b(?:def|class|return|if|elif|else|for|while|import|from|as|in|is|"
        r"not|and|or|None|True|False|self|try|except|finally|with|lambda|"
        r"yield|raise|pass|break|continue|async|await|global|nonlocal|"
        r"assert|del|new|const|let|var|function|public|private|static|void|"
        r"int|float|string|bool|struct)\b"
    )
    _STRING = re.compile(r"(\"\"\".*?\"\"\"|'''.*?'''|\"[^\"\n]*\"|'[^'\n]*')")
    _COMMENT = re.compile(r"(#[^\n]*|//[^\n]*)")
    _NUMBER = re.compile(r"\b\d+(\.\d+)?\b")
    _FUNC_DEF = re.compile(r"\b(?:def|function)\s+(\w+)")

    def __init__(self, document) -> None:
        super().__init__(document)
        self._keyword_fmt = QTextCharFormat()
        self._keyword_fmt.setForeground(QColor("#c586c0"))
        self._keyword_fmt.setFontWeight(QFont.Bold)

        self._string_fmt = QTextCharFormat()
        self._string_fmt.setForeground(QColor("#ce9178"))

        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor("#6a9955"))
        self._comment_fmt.setFontItalic(True)

        self._number_fmt = QTextCharFormat()
        self._number_fmt.setForeground(QColor("#b5cea8"))

        self._func_fmt = QTextCharFormat()
        self._func_fmt.setForeground(QColor("#dcdcaa"))

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt override)
        if not self.currentBlock().blockFormat().nonBreakableLines():
            return  # not inside a ``` code fence -- leave regular text alone

        for match in self._KEYWORDS.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._keyword_fmt)
        for match in self._NUMBER.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._number_fmt)
        for match in self._FUNC_DEF.finditer(text):
            self.setFormat(match.start(1), match.end(1) - match.start(1), self._func_fmt)
        # Strings/comments applied last so they win over any keyword-looking
        # substring that happens to appear inside one (e.g. the word "class"
        # inside a docstring).
        for match in self._STRING.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._string_fmt)
        for match in self._COMMENT.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._comment_fmt)


class AnswerStreamThread(QThread):
    """Runs the websocket client on its own asyncio loop in a background
    thread and emits each parsed message as a Qt signal."""

    message_received = Signal(dict)
    connection_status = Signal(str)

    def __init__(self, ws_url: str, session_id: str) -> None:
        super().__init__()
        self._ws_url = f"{ws_url}?session_id={session_id}"
        self._stop_event = threading.Event()

    def run(self) -> None:
        asyncio.run(self._listen())

    def stop(self) -> None:
        self._stop_event.set()

    async def _listen(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.connection_status.emit("connecting")
                async with websockets.connect(self._ws_url, max_size=None) as websocket:
                    self.connection_status.emit("connected")
                    async for raw in websocket:
                        if self._stop_event.is_set():
                            break
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self.message_received.emit(payload)
            except Exception:
                self.connection_status.emit("disconnected")
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(2)  # retry


class UploadThread(QThread):
    """Uploads a PDF (resume or JD) to POST /session/upload on a background
    thread -- multipart file upload isn't a quick fire-and-forget call like
    the JSON POSTs elsewhere (reading + sending a multi-page PDF can take a
    moment), so this reports back via a signal instead of blocking the UI
    thread or silently failing."""

    finished_signal = Signal(bool, str, str)  # success, field, message

    def __init__(self, api_url: str, session_id: str, field: str, path: str) -> None:
        super().__init__()
        self._api_url = api_url
        self._session_id = session_id
        self._field = field
        self._path = path

    def run(self) -> None:
        try:
            with open(self._path, "rb") as handle:
                response = requests.post(
                    f"{self._api_url}/session/upload",
                    data={"session_id": self._session_id, "field": self._field},
                    files={"file": (os.path.basename(self._path), handle, "application/pdf")},
                    timeout=30,
                )
            if response.ok:
                chars = response.json().get("characters_extracted", "?")
                self.finished_signal.emit(
                    True, self._field, f"uploaded ({chars} characters extracted)"
                )
            else:
                detail = response.json().get("detail", response.text)[:150]
                self.finished_signal.emit(False, self._field, detail)
        except Exception as exc:
            self.finished_signal.emit(False, self._field, str(exc)[:150])


class OverlayWindow(QWidget):
    def __init__(self, api_url: str, session_id: str, *, capturable: bool) -> None:
        super().__init__()
        self._api_url = api_url.rstrip("/")
        self._session_id = session_id
        self._capturable = capturable

        self._drag_offset: QPoint | None = None
        self._resize_edges: tuple[bool, bool, bool, bool] | None = None
        self._resize_start_geom: QRect | None = None
        self._resize_start_mouse: QPoint | None = None
        self._opacity = 0.85

        # Tokens land here as they arrive; a timer drains it at a readable
        # pace so the answer visibly streams in even when the LLM itself
        # returns most of the tokens in one fast burst.
        self._reveal_buffer = ""
        # Markdown source text revealed so far -- rebuilt into rich text via
        # setMarkdown() on a render tick, separate from _reveal_buffer
        # (not-yet-revealed) and _live_answer_text (the full raw answer).
        self._revealed_text = ""
        self._render_tick_counter = 0
        self._scroll_anim: QPropertyAnimation | None = None

        # The "live" answer -- the one currently streaming, or the most
        # recent one -- tracked separately from what's on screen, so
        # browsing older Q&A doesn't lose or corrupt it.
        self._live_question = "Waiting for a question..."
        self._live_answer_text = ""
        self._live_low_confidence = False

        # Completed Q&A pairs, oldest first. _view_index is None when showing
        # the live answer; otherwise it's an index into this list.
        self._history: list[dict] = []
        self._view_index: int | None = None

        self._model_catalog: dict[str, list[str]] = {}
        self._activity_state = "listening"
        self._activity_pulse_on = True
        self._last_pipeline_ms: int | None = None
        self._upload_threads: list[UploadThread] = []

        self._build_ui()
        self._position_default()
        self.setMouseTracking(True)
        self._load_model_catalog()

        self._reveal_timer = QTimer(self)
        self._reveal_timer.setInterval(REVEAL_TICK_MS)
        self._reveal_timer.timeout.connect(self._reveal_tick)
        self._reveal_timer.start()

        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(ACTIVITY_PULSE_MS)
        self._activity_timer.timeout.connect(self._activity_pulse_tick)
        self._activity_timer.start()
        self._set_activity("listening")

        QShortcut(QKeySequence("Alt+Left"), self, activated=self._show_previous)
        QShortcut(QKeySequence("Alt+Right"), self, activated=self._show_next)
        QShortcut(QKeySequence("Alt+L"), self, activated=self._jump_to_live)

        ws_url = self._api_url.replace("http://", "ws://").replace("https://", "wss://")
        self._stream = AnswerStreamThread(f"{ws_url}/ws/answers", session_id)
        self._stream.message_received.connect(self._on_message)
        self._stream.connection_status.connect(self._on_status)
        self._stream.start()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # keeps it out of the taskbar / alt-tab list
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(440, 360)
        self.setWindowOpacity(self._opacity)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(6)

        header = QHBoxLayout()
        self._status_label = QLabel("connecting...")
        self._status_label.setStyleSheet("color: #9adbff; font-size: 11px;")
        header.addWidget(self._status_label)
        header.addStretch()

        self._analyze_btn = QPushButton("\U0001F4F7 Analyze Screen")
        self._analyze_btn.setToolTip(
            "Capture your screen and ask the AI about it (uses the \u270e question "
            "box above as the prompt, if you've typed one)"
        )
        self._analyze_btn.setStyleSheet(
            "QPushButton { color: white; background: rgba(255,255,255,30); "
            "border: none; border-radius: 6px; font-size: 11px; padding: 3px 8px; }"
            "QPushButton:hover { background: rgba(255,255,255,70); }"
            "QPushButton:disabled { color: rgba(255,255,255,90); }"
        )
        self._analyze_btn.clicked.connect(self._on_analyze_screen_clicked)
        header.addWidget(self._analyze_btn)

        close_btn = QPushButton("x")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton { color: white; background: rgba(255,255,255,30); "
            "border: none; border-radius: 10px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(255,80,80,180); }"
        )
        close_btn.clicked.connect(self._shutdown)
        header.addWidget(close_btn)
        root.addLayout(header)

        activity_row = QHBoxLayout()
        activity_row.setSpacing(6)
        self._activity_dot = QLabel()
        self._activity_dot.setFixedSize(10, 10)
        activity_row.addWidget(self._activity_dot)
        self._activity_label = QLabel("Listening")
        self._activity_label.setStyleSheet("font-size: 11px; font-weight: 600;")
        activity_row.addWidget(self._activity_label)
        activity_row.addStretch()
        root.addLayout(activity_row)

        combo_style = (
            "QComboBox { color: #f2f2f2; background: rgba(255,255,255,25); "
            "border: 1px solid rgba(255,255,255,40); border-radius: 6px; "
            "padding: 2px 6px; font-size: 11px; }"
            "QComboBox QAbstractItemView { background: #23232b; color: #f2f2f2; "
            "selection-background-color: #3a3a46; }"
        )

        model_row = QHBoxLayout()
        self._provider_combo = QComboBox()
        self._provider_combo.setStyleSheet(combo_style)
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        model_row.addWidget(self._provider_combo, stretch=1)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)  # lets you type a model not in the catalog
        self._model_combo.setStyleSheet(combo_style)
        self._model_combo.currentTextChanged.connect(self._push_model_selection)
        model_row.addWidget(self._model_combo, stretch=2)

        refresh_btn = QPushButton("\u21bb")
        refresh_btn.setFixedSize(22, 22)
        refresh_btn.setToolTip("Refresh provider/model list from the backend")
        refresh_btn.setStyleSheet(
            "QPushButton { color: white; background: rgba(255,255,255,25); "
            "border: none; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(255,255,255,60); }"
        )
        refresh_btn.clicked.connect(lambda: self._reload_model_catalog())
        model_row.addWidget(refresh_btn)
        root.addLayout(model_row)

        upload_row = QHBoxLayout()
        upload_row.setSpacing(4)
        upload_btn_style = (
            "QPushButton { color: white; background: rgba(255,255,255,25); "
            "border: none; border-radius: 6px; font-size: 10px; padding: 3px 6px; }"
            "QPushButton:hover { background: rgba(255,255,255,60); }"
        )
        upload_resume_btn = QPushButton("\U0001F4C4 Upload Resume PDF")
        upload_resume_btn.setStyleSheet(upload_btn_style)
        upload_resume_btn.setToolTip("Upload a PDF resume -- extracted text feeds every answer")
        upload_resume_btn.clicked.connect(lambda: self._pick_and_upload_pdf("resume"))
        upload_row.addWidget(upload_resume_btn)

        upload_jd_btn = QPushButton("\U0001F4C4 Upload JD PDF")
        upload_jd_btn.setStyleSheet(upload_btn_style)
        upload_jd_btn.setToolTip("Upload a PDF job description -- extracted text feeds every answer")
        upload_jd_btn.clicked.connect(lambda: self._pick_and_upload_pdf("job_description"))
        upload_row.addWidget(upload_jd_btn)
        root.addLayout(upload_row)

        self._upload_status_label = QLabel("")
        self._upload_status_label.setStyleSheet("color: rgba(255,255,255,110); font-size: 9px;")
        root.addWidget(self._upload_status_label)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        nav_btn_style = (
            "QPushButton { color: white; background: rgba(255,255,255,25); "
            "border: none; border-radius: 6px; font-size: 11px; padding: 1px 6px; }"
            "QPushButton:hover { background: rgba(255,255,255,60); }"
            "QPushButton:disabled { color: rgba(255,255,255,60); }"
        )
        self._prev_btn = QPushButton("\u2190 Prev")
        self._prev_btn.setStyleSheet(nav_btn_style)
        self._prev_btn.setToolTip("Previous question (Alt+Left)")
        self._prev_btn.clicked.connect(self._show_previous)
        nav_row.addWidget(self._prev_btn)

        self._nav_label = QLabel("Live")
        self._nav_label.setStyleSheet("color: rgba(255,255,255,160); font-size: 10px;")
        self._nav_label.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self._nav_label, stretch=1)

        self._next_btn = QPushButton("Next \u2192")
        self._next_btn.setStyleSheet(nav_btn_style)
        self._next_btn.setToolTip("Next question (Alt+Right)")
        self._next_btn.clicked.connect(self._show_next)
        nav_row.addWidget(self._next_btn)

        self._live_btn = QPushButton("\u25cf Live")
        self._live_btn.setStyleSheet(nav_btn_style)
        self._live_btn.setToolTip("Jump back to the current question (Alt+L)")
        self._live_btn.clicked.connect(self._jump_to_live)
        nav_row.addWidget(self._live_btn)
        root.addLayout(nav_row)
        self._update_nav_buttons()

        self._question_label = QLabel("Waiting for a question...")
        self._question_label.setWordWrap(True)
        self._question_label.setStyleSheet(_QUESTION_STYLE_NORMAL)

        self._heard_label = QLabel("")
        self._heard_label.setWordWrap(True)
        self._heard_label.setStyleSheet("color: rgba(255,255,255,130); font-size: 10px;")
        root.addWidget(self._heard_label)

        self._timing_label = QLabel("")
        self._timing_label.setStyleSheet("color: rgba(255,255,255,100); font-size: 9px;")
        root.addWidget(self._timing_label)

        question_row = QHBoxLayout()
        question_row.addWidget(self._question_label, stretch=1)

        edit_btn = QPushButton("\u270e")  # pencil -- correct a misheard question
        edit_btn.setFixedSize(20, 20)
        edit_btn.setToolTip("Question misheard? Click to correct it and re-ask.")
        edit_btn.setStyleSheet(
            "QPushButton { color: white; background: rgba(255,255,255,30); "
            "border: none; border-radius: 10px; }"
            "QPushButton:hover { background: rgba(255,255,255,70); }"
        )
        edit_btn.clicked.connect(self._start_manual_edit)
        question_row.addWidget(edit_btn)
        root.addLayout(question_row)

        self._question_edit = QLineEdit()
        self._question_edit.setStyleSheet(
            "QLineEdit { color: #f2f2f2; background: rgba(255,255,255,25); "
            "border: 1px solid rgba(255,255,255,60); border-radius: 6px; "
            "padding: 3px 6px; font-size: 12px; }"
        )
        self._question_edit.setPlaceholderText("Type the real question and press Enter...")
        self._question_edit.returnPressed.connect(self._submit_manual_question)
        self._question_edit.hide()
        root.addWidget(self._question_edit)

        self._answer_view = QTextEdit()
        self._answer_view.setReadOnly(True)
        self._answer_view.setFrameStyle(0)

        self._answer_view.setCursor(Qt.ArrowCursor)
        self._answer_view.viewport().setCursor(Qt.ArrowCursor)
        
        self._answer_view.setStyleSheet(
            "QTextEdit { color: #f2f2f2; background: transparent; "
            "font-size: 14px; }"
        )
        self._answer_view.setFont(QFont("Segoe UI", 11))
        self._highlighter = _AnswerSyntaxHighlighter(self._answer_view.document())
        root.addWidget(self._answer_view, stretch=1)
        self._scroll_nav = _AnswerScrollNav(self)
        self._answer_view.viewport().installEventFilter(self._scroll_nav)

        hint = QLabel(
            "drag header to move | drag edges to resize | scroll answer: browse history | "
            "scroll elsewhere: opacity | \u270e edit question | esc: quit"
        )
        hint.setStyleSheet("color: rgba(255,255,255,110); font-size: 9px;")
        root.addWidget(hint)

    def _position_default(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 30, screen.top() + 60)

    # ---------- model picker ----------

    def _load_model_catalog(self, attempt: int = 1) -> None:
        catalog = _http_get_json(f"{self._api_url}/models", timeout=3.0)
        if catalog is None and attempt < 6:
            # The backend may still be starting up (a common race if the
            # overlay is launched right after `uvicorn` in another terminal)
            # -- retry a few times before giving up, instead of silently
            # locking in the single-provider fallback below forever.
            QTimer.singleShot(1000, lambda: self._load_model_catalog(attempt + 1))
            return

        self._model_catalog = (catalog or {}).get("catalog", {}) or {
            "groq": ["llama-3.3-70b-versatile"],
        }
        if catalog is None:
            self._status_label.setText("models: backend unreachable (showing groq only)")

        current = _http_get_json(f"{self._api_url}/session/{self._session_id}")
        current_provider = None
        current_model = None
        if current:
            context = current.get("context", {})
            current_provider = context.get("answer_provider")
            current_model = context.get("answer_model")

        self._provider_combo.blockSignals(True)
        self._provider_combo.addItems(list(self._model_catalog.keys()))
        if current_provider and current_provider in self._model_catalog:
            self._provider_combo.setCurrentText(current_provider)
        self._provider_combo.blockSignals(False)

        self._populate_models_for_provider(
            self._provider_combo.currentText(), preselect=current_model
        )

    def _populate_models_for_provider(self, provider: str, *, preselect: str | None = None) -> None:
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItems(self._model_catalog.get(provider, []))
        if preselect:
            self._model_combo.setCurrentText(preselect)
        self._model_combo.blockSignals(False)

    def _reload_model_catalog(self) -> None:
        """Manual retry (the refresh button) -- lets you refresh the provider
        list without restarting the whole overlay, e.g. after the backend
        finishes starting up, or after adding a new API key to .env and
        restarting the backend."""
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        self._provider_combo.blockSignals(False)
        self._load_model_catalog()

    def _on_provider_changed(self, provider: str) -> None:
        if not provider:
            return
        self._populate_models_for_provider(provider)
        self._push_model_selection()

    def _push_model_selection(self) -> None:
        provider = self._provider_combo.currentText()
        model = self._model_combo.currentText()
        if not provider or not model:
            return
        threading.Thread(
            target=_http_post_json,
            args=(
                f"{self._api_url}/session",
                {
                    "session_id": self._session_id,
                    "answer_provider": provider,
                    "answer_model": model,
                },
            ),
            daemon=True,
        ).start()

    # ---------- resume/JD PDF upload ----------

    def _pick_and_upload_pdf(self, field: str) -> None:
        label = "resume" if field == "resume" else "job description"
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {label} PDF", "", "PDF Files (*.pdf)"
        )
        if not path:
            return

        self._upload_status_label.setStyleSheet("color: rgba(255,255,255,150); font-size: 9px;")
        self._upload_status_label.setText(f"uploading {label}...")

        thread = UploadThread(self._api_url, self._session_id, field, path)
        thread.finished_signal.connect(self._on_upload_finished)
        thread.finished.connect(lambda: self._upload_threads.remove(thread))
        self._upload_threads.append(thread)  # keep a reference so it isn't garbage-collected
        thread.start()

    def _on_upload_finished(self, success: bool, field: str, message: str) -> None:
        label = "Resume" if field == "resume" else "Job description"
        if success:
            self._upload_status_label.setStyleSheet("color: #7CFC9E; font-size: 9px;")
            self._upload_status_label.setText(f"{label}: {message}")
        else:
            self._upload_status_label.setStyleSheet("color: #ff8a8a; font-size: 9px;")
            self._upload_status_label.setText(f"{label} upload failed: {message}")

    # ---------- manual question correction ----------

    def _start_manual_edit(self) -> None:
        current = self._question_label.text()
        if current == "Waiting for a question...":
            current = ""
        self._question_edit.setText(current)
        self._question_label.hide()
        self._question_edit.show()
        self._question_edit.setFocus()
        self._question_edit.selectAll()

    def _submit_manual_question(self) -> None:
        text = self._question_edit.text().strip()
        self._question_edit.hide()
        self._question_label.show()
        if not text:
            return

        # Optimistic UI: show what we're about to ask right away rather than
        # waiting for the round trip, since this is a deliberate correction.
        self._question_label.setStyleSheet(_QUESTION_STYLE_NORMAL)
        self._question_label.setText(text)
        self._heard_label.setText("re-asking with corrected question...")
        self._reveal_buffer = ""
        self._answer_view.clear()

        threading.Thread(
            target=_http_post_json,
            args=(
                f"{self._api_url}/ask",
                {"session_id": self._session_id, "question": text},
            ),
            daemon=True,
        ).start()

    # ---------- screen analysis ("Analyze Screen" button) ----------

    def _on_analyze_screen_clicked(self) -> None:
        # Whatever's currently typed in the (normally hidden) question-edit
        # box doubles as the guiding question for screen analysis -- e.g.
        # type "what does this error mean?" then click Analyze Screen. If
        # it's empty/hidden, a generic prompt is used server-side instead.
        guiding_question = self._question_edit.text().strip()

        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setText("Capturing...")
        self._heard_label.setStyleSheet("color: rgba(255,255,255,130); font-size: 10px;")
        self._heard_label.setText("capturing screen...")

        # Hide the overlay itself before capturing so it doesn't end up in
        # its own screenshot -- the Windows capture-exclusion flag (see
        # showEvent below) only fools *other* apps' screen-share/recording
        # APIs, not a same-process screen grab like this, and isn't applied
        # at all on macOS/Linux. A short delay after hide() gives the window
        # manager time to actually remove it from the screen before we grab.
        self.hide()
        QTimer.singleShot(200, lambda: self._capture_and_send_screen(guiding_question))

    def _capture_and_send_screen(self, guiding_question: str) -> None:
        image_b64, media_type = self._grab_screen_base64()
        self.show()
        self._analyze_btn.setEnabled(True)
        self._analyze_btn.setText("\U0001F4F7 Analyze Screen")

        if not image_b64:
            self._heard_label.setStyleSheet("color: #ff8a8a; font-size: 10px;")
            self._heard_label.setText("screen capture failed")
            return

        self._heard_label.setStyleSheet("color: rgba(255,255,255,130); font-size: 10px;")
        self._heard_label.setText("analyzing screen...")
        self._set_activity("thinking")

        threading.Thread(
            target=_http_post_json,
            args=(
                f"{self._api_url}/analyze-screen",
                {
                    "session_id": self._session_id,
                    "image_base64": image_b64,
                    "media_type": media_type,
                    "question": guiding_question or None,
                },
            ),
            daemon=True,
        ).start()

    def _grab_screen_base64(self) -> tuple[str, str] | tuple[None, None]:
        """Grabs the primary screen, downscales it, and returns
        (base64_png_or_jpeg, media_type) -- or (None, None) if capture
        failed. Deliberately NOT a full-resolution lossless PNG: on a call
        where you're also screen-sharing + on camera, that upload (often
        2-5MB for a full HD screenshot) competes for the same uplink
        bandwidth as the live audio going to Groq's STT API, and was
        stalling those small audio requests past their timeout. Downscaling
        to a capped width and encoding as JPEG cuts the payload by roughly
        10-20x with no real loss in readability for code/text on screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return None, None
        try:
            pixmap = screen.grabWindow(0)
            if pixmap.isNull():
                return None, None

            max_width = 1600
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)

            buffer = QBuffer()
            buffer.open(QIODevice.WriteOnly)
            if not pixmap.save(buffer, "JPEG", 82):
                return None, None
            data = bytes(buffer.data())
            buffer.close()
            return base64.b64encode(data).decode("ascii"), "image/jpeg"
        except Exception:
            return None, None

    # ---------- painting (rounded translucent panel) ----------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 14, 14)
        painter.fillPath(path, QColor(20, 20, 25, 205))

    # ---------- window chrome: drag to move, drag edges to resize, scroll to fade, Esc to quit ----------

    def _edges_at(self, pos: QPoint) -> tuple[bool, bool, bool, bool]:
        """Returns (left, top, right, bottom) -- which edges/corners pos is near."""
        rect = self.rect()
        left = pos.x() <= RESIZE_MARGIN
        right = pos.x() >= rect.width() - RESIZE_MARGIN
        top = pos.y() <= RESIZE_MARGIN
        bottom = pos.y() >= rect.height() - RESIZE_MARGIN
        return left, top, right, bottom

    def _update_cursor_for_edges(self, edges: tuple[bool, bool, bool, bool]) -> None:
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        edges = self._edges_at(pos)
        if any(edges):
            self._resize_edges = edges
            self._resize_start_geom = self.geometry()
            self._resize_start_mouse = event.globalPosition().toPoint()
        elif pos.y() <= 34:  # only drag-to-move from the header strip
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._resize_edges is not None and (event.buttons() & Qt.LeftButton):
            self._perform_resize(event.globalPosition().toPoint())
            return
        if self._drag_offset is not None and (event.buttons() & Qt.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            return
        # not dragging: just update the cursor to hint resize/move affordances
        self._update_cursor_for_edges(self._edges_at(event.position().toPoint()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None
        self._resize_edges = None
        self._resize_start_geom = None
        self._resize_start_mouse = None

    def _perform_resize(self, global_pos: QPoint) -> None:
        left, top, right, bottom = self._resize_edges
        delta = global_pos - self._resize_start_mouse
        geom = QRect(self._resize_start_geom)

        if left:
            geom.setLeft(geom.left() + delta.x())
        if right:
            geom.setRight(geom.right() + delta.x())
        if top:
            geom.setTop(geom.top() + delta.y())
        if bottom:
            geom.setBottom(geom.bottom() + delta.y())

        if geom.width() < MIN_WIDTH:
            if left:
                geom.setLeft(geom.right() - MIN_WIDTH)
            else:
                geom.setRight(geom.left() + MIN_WIDTH)
        if geom.height() < MIN_HEIGHT:
            if top:
                geom.setTop(geom.bottom() - MIN_HEIGHT)
            else:
                geom.setBottom(geom.top() + MIN_HEIGHT)

        self.setGeometry(geom)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = 0.05 if event.angleDelta().y() > 0 else -0.05
        self._opacity = min(1.0, max(0.25, self._opacity + delta))
        self.setWindowOpacity(self._opacity)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._shutdown()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stream.stop()
        event.accept()

    def _shutdown(self) -> None:
        self._stream.stop()
        QApplication.instance().quit()

    # ---------- screen-capture exclusion (Windows only) ----------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if sys.platform == "win32" and not self._capturable:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

    # ---------- typewriter-paced reveal ----------

    def _reveal_tick(self) -> None:
        if not self._reveal_buffer:
            return

        # Reveal more per tick when the backlog is growing (e.g. the LLM
        # finished generating well before we've finished displaying it), so
        # we visibly stream without ever falling permanently behind.
        base_chunk = max(1, round(REVEAL_CHARS_PER_SECOND * REVEAL_TICK_MS / 1000))
        backlog_chunk = max(0, len(self._reveal_buffer) - 80) // 3
        chunk_size = base_chunk + backlog_chunk

        chunk, self._reveal_buffer = (
            self._reveal_buffer[:chunk_size],
            self._reveal_buffer[chunk_size:],
        )
        if self._view_index is None:
            self._revealed_text += chunk
            self._render_tick_counter += 1
            # Re-render the markdown view (full reparse + re-highlight) every
            # ~4th reveal tick (~60ms) rather than every single one (15ms) --
            # that's still visually fluid but a lot lighter on CPU than
            # reparsing the whole document 66 times a second. Always render
            # on the tick that empties the buffer so the final state is never
            # left stale by the modulo skip.
            if self._render_tick_counter % 4 == 0 or not self._reveal_buffer:
                self._render_live_answer()

    def _flush_reveal_buffer(self) -> None:
        if self._reveal_buffer:
            if self._view_index is None:
                self._revealed_text += self._reveal_buffer
                self._render_live_answer()
            self._reveal_buffer = ""

    def _render_live_answer(self) -> None:
        """Re-renders _revealed_text as markdown (code fences -> monospace +
        syntax-highlighted blocks, bold/headers -> rich text) and keeps the
        view pinned to the bottom with a short eased scroll instead of an
        instant jump -- setMarkdown() resets the scrollbar to 0 as a side
        effect of replacing the document, so the previous position is
        restored in the same call (before Qt repaints) to avoid a visible
        flash back to the top on every render tick."""
        bar = self._answer_view.verticalScrollBar()
        previous_value = bar.value()
        was_at_bottom = previous_value >= bar.maximum() - 4

        self._answer_view.setMarkdown(self._revealed_text)
        self._style_code_blocks()

        bar.setValue(previous_value)
        if was_at_bottom:
            self._smooth_scroll_to_bottom()

    def _style_code_blocks(self) -> None:
        """setMarkdown() renders fenced code blocks in a monospace font but
        with no visual separation from regular text -- this gives them a
        faint background panel (like a code editor) so they actually read
        as distinct blocks, using the same NonBreakableLines marker the
        syntax highlighter keys off of."""
        doc = self._answer_view.document()
        block = doc.begin()
        while block.isValid():
            block_format = block.blockFormat()
            if block_format.nonBreakableLines():
                new_format = QTextBlockFormat(block_format)
                new_format.setBackground(QColor(255, 255, 255, 18))
                new_format.setLeftMargin(10)
                new_format.setRightMargin(6)
                new_format.setTopMargin(2)
                new_format.setBottomMargin(2)
                cursor = QTextCursor(block)
                cursor.setBlockFormat(new_format)
            block = block.next()

    def _smooth_scroll_to_bottom(self) -> None:
        """Animates the scrollbar down to the bottom over ~180ms with an
        ease-out curve, instead of _reveal_tick's old behavior of snapping
        straight to verticalScrollBar().maximum() every tick -- that's what
        looked like it was 'jumping to the next page' as soon as content
        overflowed the visible area. Restarting a short animation on each
        render tick (rather than one long one) makes it track a continuously
        growing document smoothly, the same way a chat app's feed glides
        down as new messages arrive."""
        bar = self._answer_view.verticalScrollBar()
        target = bar.maximum()
        if target <= bar.value():
            return
        if self._scroll_anim is not None:
            self._scroll_anim.stop()
            self._scroll_anim.deleteLater()
        anim = QPropertyAnimation(bar, b"value", self)
        anim.setDuration(180)
        anim.setStartValue(bar.value())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._scroll_anim = anim

    # ---------- Q&A history navigation ----------

    def _update_nav_buttons(self) -> None:
        total = len(self._history)
        if self._view_index is None:
            self._nav_label.setText(f"Live" + (f"  ({total} past)" if total else ""))
            self._prev_btn.setEnabled(total > 0)
            self._next_btn.setEnabled(False)
            self._live_btn.setEnabled(False)
        else:
            self._nav_label.setText(f"{self._view_index + 1} / {total}")
            self._prev_btn.setEnabled(self._view_index > 0)
            self._next_btn.setEnabled(True)
            self._live_btn.setEnabled(True)

    def _show_previous(self) -> None:
        if not self._history:
            return
        if self._view_index is None:
            self._view_index = len(self._history) - 1
        elif self._view_index > 0:
            self._view_index -= 1
        self._render_history_item()

    def _show_next(self) -> None:
        if self._view_index is None:
            return
        if self._view_index < len(self._history) - 1:
            self._view_index += 1
            self._render_history_item()
        else:
            self._jump_to_live()

    def _jump_to_live(self) -> None:
        self._view_index = None
        self._render_question(self._live_question, self._live_low_confidence)
        self._revealed_text = self._live_answer_text
        self._answer_view.setMarkdown(self._revealed_text)
        self._style_code_blocks()
        bar = self._answer_view.verticalScrollBar()
        bar.setValue(bar.maximum())  # deliberate instant jump, not animated
        self._reveal_buffer = ""  # already fully caught up visually
        self._update_nav_buttons()

    def _render_history_item(self) -> None:
        if self._view_index is None or not (0 <= self._view_index < len(self._history)):
            return
        item = self._history[self._view_index]
        self._render_question(item["question"], item["low_confidence"])
        self._answer_view.setMarkdown(item["answer"])
        self._style_code_blocks()
        self._update_nav_buttons()

    def _render_question(self, question: str, low_confidence: bool) -> None:
        if low_confidence:
            self._question_label.setStyleSheet(_QUESTION_STYLE_LOW_CONFIDENCE)
            self._question_label.setText(f"\u26a0 {question}")
        else:
            self._question_label.setStyleSheet(_QUESTION_STYLE_NORMAL)
            self._question_label.setText(question)

    # ---------- activity indicator (Listening / Hearing / Thinking / Answering) ----------

    def _set_activity(self, state: str) -> None:
        if state not in _ACTIVITY_STATES:
            return
        self._activity_state = state
        self._activity_pulse_on = True
        text, color = _ACTIVITY_STATES[state]
        self._activity_label.setText(text)
        self._activity_label.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")
        self._render_activity_dot(color, on=True)

    def _activity_pulse_tick(self) -> None:
        self._activity_pulse_on = not self._activity_pulse_on
        _, color = _ACTIVITY_STATES[self._activity_state]
        self._render_activity_dot(color, on=self._activity_pulse_on)

    def _render_activity_dot(self, color: str, *, on: bool) -> None:
        opacity = 255 if on else 90
        rgba = QColor(color)
        self._activity_dot.setStyleSheet(
            f"background-color: rgba({rgba.red()},{rgba.green()},{rgba.blue()},{opacity}); "
            "border-radius: 5px;"
        )

    # ---------- websocket message handling ----------

    def _on_status(self, status: str) -> None:
        colors = {
            "connecting": "#9adbff",
            "connected": "#7CFC9E",
            "disconnected": "#ff8080",
        }
        self._status_label.setStyleSheet(
            f"color: {colors.get(status, '#9adbff')}; font-size: 11px;"
        )
        self._status_label.setText(status)

    def _on_message(self, payload: dict) -> None:
        message_type = payload.get("type")

        if message_type == "transcript":
            # Live "what we actually heard" feed -- lets you catch a
            # mishearing before/while an answer generates, independent of
            # whether the gate decides to answer it.
            self._set_activity("hearing")
            text = payload.get("text", "")
            confidence = payload.get("confidence")
            if payload.get("low_confidence"):
                self._heard_label.setStyleSheet("color: #ff8a8a; font-size: 10px;")
                self._heard_label.setText(f"heard (uncertain): {text}")
            else:
                self._heard_label.setStyleSheet("color: rgba(255,255,255,130); font-size: 10px;")
                label = f"heard: {text}"
                if confidence is not None:
                    label += f"  ({int(confidence * 100)}%)"
                self._heard_label.setText(label)
        elif message_type == "question_gate":
            if payload.get("should_answer"):
                self._set_activity("thinking")
            else:
                # Wasn't actually a question (small talk, still mid-sentence,
                # etc.) -- go back to a plain listening state.
                self._set_activity("listening")
        elif message_type == "answer_start":
            self._set_activity("answering")
            self._reveal_buffer = ""
            self._revealed_text = ""
            self._render_tick_counter = 0
            question = payload.get("question", "")
            self._live_question = question
            self._live_answer_text = ""
            self._live_low_confidence = bool(payload.get("low_confidence"))
            self._last_pipeline_ms = payload.get("pipeline_ms")
            self._timing_label.setText(
                f"heard\u2192answer: {self._last_pipeline_ms}ms + LLM: ..."
                if self._last_pipeline_ms is not None
                else ""
            )
            if self._view_index is None:
                self._render_question(question, self._live_low_confidence)
                self._answer_view.clear()
            else:
                # Browsing history when a new answer starts -- don't yank the
                # view away, just flag that something new has arrived.
                self._update_nav_buttons()
        elif message_type == "answer_token":
            self._set_activity("answering")
            token = payload.get("token", "")
            self._reveal_buffer += token
            self._live_answer_text += token
            first_token_ms = payload.get("first_token_latency_ms")
            if first_token_ms is not None and self._last_pipeline_ms is not None:
                total = self._last_pipeline_ms + first_token_ms
                self._timing_label.setText(
                    f"heard\u2192answer: {self._last_pipeline_ms}ms + LLM: {first_token_ms}ms "
                    f"= {total}ms to first word"
                )
        elif message_type == "answer_done":
            self._set_activity("listening")
            answer_text = payload.get("answer", "").strip()
            if answer_text and self._live_question:
                self._history.append(
                    {
                        "question": self._live_question,
                        "answer": answer_text,
                        "low_confidence": self._live_low_confidence,
                    }
                )
                del self._history[:-100]  # cap memory: keep the most recent 100
            self._update_nav_buttons()
        elif message_type == "answer_cancelled":
            self._set_activity("listening")
            self._flush_reveal_buffer()
            if self._view_index is None:
                self._answer_view.append("\n[cancelled]")
        elif message_type == "error":
            self._set_activity("listening")
            self._flush_reveal_buffer()
            if self._view_index is None:
                self._answer_view.append(f"\n[error] {payload.get('message', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InterviewCopilot floating overlay.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--session-id", default="default")
    parser.add_argument(
        "--capturable",
        action="store_true",
        help="Do NOT exclude the overlay from screen capture (visible in screen share/recording).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = OverlayWindow(args.api_url, args.session_id, capturable=args.capturable)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
