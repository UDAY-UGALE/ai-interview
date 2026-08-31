"""The "Project & Experience Context" dialog.

Why this is a separate window rather than more rows in the overlay: the
overlay is a 440px panel that floats over a live video call, and it earns
its size by showing one question and one answer. Twelve fields per project
does not belong there. This opens from a button, is filled in before the
interview (or during, if the interviewer asks about something you had not
written down), saves, and closes.

What it is FOR is worth stating plainly, because the UI cannot: a resume
says "worked on a real-time AI application", and no answer to "why did you
pick Deepgram over Whisper?" can be built from that. Whatever is typed or
uploaded here is what the model is allowed to claim you personally did --
and, just as importantly, everything it is NOT allowed to claim.

Two ways in, both first-class:

* structured projects -- name, role, architecture, decisions, trade-offs,
  metrics, one card per project;
* paste or upload -- PDF, DOCX or TXT of notes you already wrote.

Documents are extracted server-side at upload time (never during a
question) and kept in the notes with a `[document: name]` marker, which is
what makes the Remove button still work the next time the dialog opens.
"""

from __future__ import annotations

import os
import re
import sys
import urllib.parse

import requests
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# Each project field: (attribute, label, placeholder, rows). Rows of 0 means
# a single-line box. The placeholders are doing real work -- they are the
# only place the UI can say "this is what a useful answer looks like", and
# an empty box labelled "Trade-offs" gets left empty.
PROJECT_FIELDS: tuple[tuple[str, str, str, int], ...] = (
    ("name", "Project Name", "Real-Time AI Interview Assistant", 0),
    (
        "description",
        "Project Description",
        "What the system does, in a few sentences. Example: captures microphone and "
        "system audio, segments speech, converts it to text, detects interview "
        "questions and generates answers with an LLM.",
        3,
    ),
    (
        "role",
        "My Role",
        "What YOU personally built, as opposed to what the team built. Example: I "
        "designed the audio pipeline and wrote the VAD and question-detection logic.",
        2,
    ),
    (
        "responsibilities",
        "My Responsibilities",
        "What you owned day to day -- design, implementation, review, deployment, "
        "on-call.",
        2,
    ),
    (
        "architecture",
        "Architecture",
        "Audio Capture -> WebSocket -> VAD -> Speech Segmenter -> STT -> Question "
        "Gate -> LLM -> Answer. Add a line on what each part does.",
        3,
    ),
    (
        "technologies",
        "Technologies",
        "Only what you actually used. Example: Python, FastAPI, WebSocket, Deepgram, "
        "Whisper, Groq, PostgreSQL, Redis, Docker.",
        2,
    ),
    (
        "challenges",
        "Challenges",
        "The hard part. Example: the system sometimes answered before the "
        "interviewer had finished speaking.",
        2,
    ),
    (
        "solutions",
        "Solutions",
        "What you did about it. Example: added VAD and a question gate so a complete "
        "speech segment has to land before the LLM is called.",
        2,
    ),
    (
        "decisions",
        "Technical Decisions",
        "Why X and not Y -- interviewers ask this constantly. Example: we ran Whisper "
        "through Groq first, then evaluated Deepgram because streaming needed lower "
        "latency.",
        2,
    ),
    (
        "tradeoffs",
        "Trade-offs",
        "What the decision cost. Example: the question gate cuts unnecessary LLM "
        "calls, but a strict gate rejects short valid questions, so it is a hybrid.",
        2,
    ),
    (
        "metrics",
        "Results / Metrics",
        "Only real measurements -- latency, accuracy, throughput, cost, users, "
        "dataset size. Anything typed here may be quoted in an answer; anything not "
        "typed here will never be invented.",
        2,
    ),
    ("additional_notes", "Additional Notes", "Anything else worth having to hand.", 2),
)

_DOCUMENT_MARKER = re.compile(r"^\[document: (.+?)\]$", re.MULTILINE)

_SUPPORTED_FILTER = (
    "Documents (*.pdf *.docx *.txt *.md);;PDF (*.pdf);;Word (*.docx);;Text (*.txt *.md)"
)


# ---------------------------------------------------------------------------
# Documents inside the notes field
# ---------------------------------------------------------------------------


def split_documents(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split stored notes into (typed text, [(filename, text), ...]).

    The marker is what makes an uploaded document a thing the user can
    still see and remove a week later, instead of text that silently fused
    into a blob on save. It is plain and readable on purpose -- the model
    sees it too, and "this came from architecture-notes.pdf" is useful to
    it rather than noise.
    """
    text = text or ""
    matches = list(_DOCUMENT_MARKER.finditer(text))
    if not matches:
        return text.strip(), []

    manual = text[: matches[0].start()].strip()
    documents: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        documents.append((match.group(1), text[match.end() : end].strip()))
    return manual, documents


def compose_documents(manual: str, documents: list[tuple[str, str]]) -> str:
    parts = [(manual or "").strip()]
    parts += [f"[document: {name}]\n{body.strip()}" for name, body in documents if body.strip()]
    return "\n\n".join(part for part in parts if part).strip()


# ---------------------------------------------------------------------------
# Network, off the UI thread
# ---------------------------------------------------------------------------


class _ApiThread(QThread):
    """One request, reported back as (ok, payload_or_message).

    Everything here is a user action with a visible result -- load, upload,
    save -- so each one gets a thread and a status line rather than
    blocking the window or failing silently.
    """

    done = Signal(bool, object)

    def __init__(self, kind: str, **kwargs) -> None:
        super().__init__()
        self._kind = kind
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            handler = getattr(self, f"_run_{self._kind}")
            self.done.emit(True, handler(**self._kwargs))
        except requests.HTTPError as exc:
            self.done.emit(False, _http_error_message(exc))
        except Exception as exc:
            self.done.emit(False, str(exc)[:200] or type(exc).__name__)

    def _run_load(self, *, url: str, headers: dict) -> dict:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def _run_extract(self, *, url: str, headers: dict, path: str) -> dict:
        with open(path, "rb") as handle:
            response = requests.post(
                url,
                files={"file": (os.path.basename(path), handle)},
                headers=headers,
                timeout=60,
            )
        response.raise_for_status()
        return response.json()

    def _run_save(self, *, url: str, headers: dict, payload: dict) -> dict:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()


def _http_error_message(exc: requests.HTTPError) -> str:
    """The server's own message, which is written to be read by a person --
    "that PDF is a scan with no text layer" beats "400 Bad Request"."""
    response = exc.response
    if response is None:
        return str(exc)[:200]
    try:
        detail = response.json().get("detail")
    except Exception:
        detail = None
    return str(detail or response.text or exc)[:300]


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

_INPUT_STYLE = (
    "QLineEdit, QTextEdit { color: #f2f2f2; background: rgba(255,255,255,18); "
    "border: 1px solid rgba(255,255,255,45); border-radius: 6px; padding: 4px 6px; "
    "font-size: 12px; }"
)
_BUTTON_STYLE = (
    "QPushButton { color: white; background: rgba(255,255,255,28); border: none; "
    "border-radius: 6px; font-size: 11px; padding: 5px 10px; }"
    "QPushButton:hover { background: rgba(255,255,255,70); }"
    "QPushButton:disabled { color: rgba(255,255,255,90); }"
)
_PRIMARY_BUTTON_STYLE = (
    "QPushButton { color: white; background: rgba(80,170,120,170); border: none; "
    "border-radius: 6px; font-size: 12px; font-weight: 600; padding: 6px 16px; }"
    "QPushButton:hover { background: rgba(90,195,135,200); }"
    "QPushButton:disabled { background: rgba(255,255,255,25); "
    "color: rgba(255,255,255,90); }"
)
_HELP_STYLE = "color: rgba(255,255,255,150); font-size: 11px;"
_LABEL_STYLE = "color: #9adbff; font-size: 11px; font-weight: 600;"


class _ProjectCard(QFrame):
    """One project. Collapsible, because four filled-in projects is a very
    long scroll and the header alone is enough to find the right one."""

    removed = Signal(object)
    changed = Signal()

    def __init__(self, data: dict | None = None, *, expanded: bool = True) -> None:
        super().__init__()
        self.setStyleSheet(
            "QFrame { background: rgba(255,255,255,12); border: 1px solid "
            "rgba(255,255,255,30); border-radius: 8px; }"
        )
        self._editors: dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self._toggle = QPushButton()
        self._toggle.setStyleSheet(
            "QPushButton { color: #ffd479; background: transparent; border: none; "
            "font-size: 12px; font-weight: 600; text-align: left; }"
        )
        self._toggle.clicked.connect(self._toggle_body)
        header.addWidget(self._toggle, stretch=1)

        remove = QPushButton("Remove")
        remove.setStyleSheet(_BUTTON_STYLE)
        remove.clicked.connect(lambda: self.removed.emit(self))
        header.addWidget(remove)
        outer.addLayout(header)

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)

        for attribute, label, placeholder, rows in PROJECT_FIELDS:
            caption = QLabel(label)
            caption.setStyleSheet(_LABEL_STYLE)
            body_layout.addWidget(caption)

            if rows:
                editor: QWidget = QTextEdit()
                editor.setAcceptRichText(False)
                editor.setPlaceholderText(placeholder)
                editor.setFixedHeight(22 * rows + 12)
                editor.textChanged.connect(self._on_changed)
            else:
                editor = QLineEdit()
                editor.setPlaceholderText(placeholder)
                editor.textChanged.connect(self._on_changed)
            editor.setStyleSheet(_INPUT_STYLE)
            body_layout.addWidget(editor)
            self._editors[attribute] = editor

        outer.addWidget(self._body)

        if data:
            self.set_values(data)
        self._body.setVisible(expanded)
        self._refresh_header()

    def _toggle_body(self) -> None:
        self._body.setVisible(not self._body.isVisible())
        self._refresh_header()

    def _on_changed(self) -> None:
        self._refresh_header()
        self.changed.emit()

    def _refresh_header(self) -> None:
        arrow = "▾" if self._body.isVisible() else "▸"
        name = self.values().get("name", "").strip() or "New project"
        self._toggle.setText(f"{arrow}  {name}")

    def set_values(self, data: dict) -> None:
        for attribute, editor in self._editors.items():
            value = str(data.get(attribute, "") or "")
            if isinstance(editor, QTextEdit):
                editor.setPlainText(value)
            else:
                editor.setText(value)

    def values(self) -> dict:
        result = {}
        for attribute, editor in self._editors.items():
            text = (
                editor.toPlainText() if isinstance(editor, QTextEdit) else editor.text()
            )
            result[attribute] = text.strip()
        return result

    def is_empty(self) -> bool:
        values = self.values()
        return not any(value for key, value in values.items() if key != "name")


class _DocumentRow(QFrame):
    """Filename, extraction status, Remove -- the three things the user
    needs to know a document actually landed."""

    removed = Signal(object)

    def __init__(self, filename: str, characters: int, *, note: str = "") -> None:
        super().__init__()
        self.filename = filename
        self.setStyleSheet("QFrame { border: none; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        name = QLabel(f"\U0001F4C4 {filename}")
        name.setStyleSheet("color: #f2f2f2; font-size: 11px;")
        row.addWidget(name, stretch=1)

        status = QLabel(note or f"{characters:,} characters extracted")
        status.setStyleSheet("color: #7CFC9E; font-size: 10px;")
        row.addWidget(status)

        remove = QPushButton("Remove")
        remove.setStyleSheet(_BUTTON_STYLE)
        remove.clicked.connect(lambda: self.removed.emit(self))
        row.addWidget(remove)


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------


class ProjectContextDialog(QDialog):
    saved = Signal(str)  # human-readable status for the overlay's status line

    def __init__(
        self,
        api_url: str,
        session_id: str,
        auth_headers: dict,
        *,
        hidden_from_capture: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._api_url = api_url.rstrip("/")
        self._session_id = session_id
        self._headers = dict(auth_headers or {})
        self._hidden_from_capture = hidden_from_capture
        self._threads: list[_ApiThread] = []
        self._documents: list[tuple[str, str]] = []  # (filename, extracted text)
        self._cards: list[_ProjectCard] = []

        self.setWindowTitle("Project & Experience Context")
        self.setMinimumSize(620, 620)
        self.setStyleSheet("QDialog { background: #1c1c22; }")

        # Closing this window must never end the interview.
        #
        # Qt quits the application when the last window with WA_QuitOnClose
        # closes -- and the overlay does NOT have that attribute, because Qt
        # clears it for Qt.Tool windows. An ordinary dialog has it set, so
        # this window was the only one Qt was counting: closing it, saved or
        # not, dropped the count to zero and took the overlay down with it,
        # mid-interview. The overlay owns its own lifetime through
        # _shutdown() (the X button and Esc); nothing here should.
        self.setAttribute(Qt.WA_QuitOnClose, False)

        self._build_ui()
        self._load_existing()

    # ---------- layout ----------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        title = QLabel("PROJECT & EXPERIENCE CONTEXT")
        title.setStyleSheet("color: #ffd479; font-size: 14px; font-weight: 700;")
        root.addWidget(title)

        blurb = QLabel(
            "Give the AI more context about your real work. Add project details, "
            "architecture, challenges, decisions and achievements so answers are "
            "based on your experience instead of assumptions.\n"
            "Everything here is sent as interview context with every question. "
            "Anything you do not write here -- a tool, a number, an incident -- "
            "will never be claimed as yours."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(_HELP_STYLE)
        root.addWidget(blurb)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid rgba(255,255,255,30); "
            "border-radius: 6px; }"
            "QTabBar::tab { color: #d8d8d8; background: rgba(255,255,255,15); "
            "padding: 6px 14px; margin-right: 2px; border-top-left-radius: 6px; "
            "border-top-right-radius: 6px; font-size: 11px; }"
            "QTabBar::tab:selected { background: rgba(255,255,255,45); color: white; }"
        )
        tabs.addTab(self._build_projects_tab(), "Projects")
        tabs.addTab(self._build_notes_tab(), "Notes & Documents")
        tabs.addTab(self._build_stories_tab(), "Interview Stories")
        root.addWidget(tabs, stretch=1)

        self._size_label = QLabel("")
        self._size_label.setWordWrap(True)
        self._size_label.setStyleSheet("color: rgba(255,255,255,120); font-size: 10px;")
        root.addWidget(self._size_label)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: rgba(255,255,255,150); font-size: 11px;")
        root.addWidget(self._status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Close")
        cancel.setStyleSheet(_BUTTON_STYLE)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)

        self._save_btn = QPushButton("Save context")
        self._save_btn.setStyleSheet(_PRIMARY_BUTTON_STYLE)
        self._save_btn.clicked.connect(self._save)
        buttons.addWidget(self._save_btn)
        root.addLayout(buttons)

    def _build_projects_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hint = QLabel(
            "One card per project. Fill in only what you have -- a project with just "
            "a description and your role is already far more than a resume line."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_HELP_STYLE)
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        self._cards_layout = QVBoxLayout(holder)
        self._cards_layout.setContentsMargins(0, 0, 6, 0)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch()
        scroll.setWidget(holder)
        layout.addWidget(scroll, stretch=1)

        add = QPushButton("+ Add Project")
        add.setStyleSheet(_BUTTON_STYLE)
        add.clicked.connect(lambda: self._add_card())
        layout.addWidget(add)
        return page

    def _build_notes_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hint = QLabel(
            "Upload project notes, technical documentation, or experience notes "
            "(PDF, DOCX, TXT) -- or just paste them below. Use this if you already "
            "have your notes written down and would rather not fill in the form."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_HELP_STYLE)
        layout.addWidget(hint)

        upload_row = QHBoxLayout()
        self._upload_btn = QPushButton("\U0001F4C4  Upload PDF / DOCX / TXT")
        self._upload_btn.setStyleSheet(_BUTTON_STYLE)
        self._upload_btn.clicked.connect(self._pick_document)
        upload_row.addWidget(self._upload_btn)
        upload_row.addStretch()
        layout.addLayout(upload_row)

        self._documents_layout = QVBoxLayout()
        self._documents_layout.setSpacing(4)
        layout.addLayout(self._documents_layout)

        paste_label = QLabel("Paste text")
        paste_label.setStyleSheet(_LABEL_STYLE)
        layout.addWidget(paste_label)

        self._notes_edit = QTextEdit()
        self._notes_edit.setAcceptRichText(False)
        self._notes_edit.setStyleSheet(_INPUT_STYLE)
        self._notes_edit.setPlaceholderText(
            "Anything that did not fit the project form: what you owned, tools you "
            "actually used, decisions and why, achievements, measurements you really "
            "have."
        )
        self._notes_edit.textChanged.connect(self._refresh_size)
        layout.addWidget(self._notes_edit, stretch=1)
        return page

    def _build_stories_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hint = QLabel(
            "Short stories for behavioural questions -- a difficult bug, a production "
            "incident, a performance fix, an architecture decision, a disagreement, "
            "leading something, a failure and what you learned. A few lines each is "
            "enough; the AI turns them into a spoken answer."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_HELP_STYLE)
        layout.addWidget(hint)

        self._stories_edit = QTextEdit()
        self._stories_edit.setAcceptRichText(False)
        self._stories_edit.setStyleSheet(_INPUT_STYLE)
        self._stories_edit.setPlaceholderText(
            "Difficult bug: answers were firing before the interviewer finished. "
            "Traced it to the VAD closing a segment on a mid-sentence pause; added an "
            "utterance-level grouping so a pause no longer ends the question."
        )
        self._stories_edit.textChanged.connect(self._refresh_size)
        layout.addWidget(self._stories_edit, stretch=1)
        return page

    # ---------- capture exclusion ----------

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._hidden_from_capture and sys.platform == "win32":
            # Same treatment as the overlay itself: this window has the
            # candidate's project notes in it, and it is opened on a machine
            # that is very often already screen-sharing.
            try:
                import ctypes

                ctypes.windll.user32.SetWindowDisplayAffinity(
                    int(self.winId()), 0x00000011  # WDA_EXCLUDEFROMCAPTURE
                )
            except Exception:
                pass

    # ---------- projects ----------

    def _add_card(self, data: dict | None = None, *, expanded: bool = True) -> _ProjectCard:
        card = _ProjectCard(data, expanded=expanded)
        card.removed.connect(self._remove_card)
        card.changed.connect(self._refresh_size)
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
        self._cards.append(card)
        self._refresh_size()
        return card

    def _remove_card(self, card: _ProjectCard) -> None:
        if card in self._cards:
            self._cards.remove(card)
        card.setParent(None)
        card.deleteLater()
        self._refresh_size()

    # ---------- documents ----------

    def _pick_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select project notes", "", _SUPPORTED_FILTER
        )
        if not path:
            return
        name = os.path.basename(path)
        if any(existing == name for existing, _text in self._documents):
            self._set_status(f"{name} has already been added.", ok=False)
            return

        self._upload_btn.setEnabled(False)
        self._set_status(f"reading {name}...")
        self._run(
            "extract",
            lambda ok, payload: self._on_extracted(ok, payload),
            url=f"{self._api_url}/session/extract",
            headers=self._headers,
            path=path,
        )

    def _on_extracted(self, ok: bool, payload) -> None:
        self._upload_btn.setEnabled(True)
        if not ok:
            self._set_status(str(payload), ok=False)
            return

        name = payload.get("filename", "document")
        text = payload.get("text", "")
        characters = payload.get("characters_extracted", len(text))
        note = None
        if payload.get("truncated"):
            note = f"{characters:,} characters (trimmed -- document was very long)"

        self._documents.append((name, text))
        self._add_document_row(name, characters, note=note or "")
        self._set_status(f"{name}: {characters:,} characters extracted. Not saved yet.")
        self._refresh_size()

    def _add_document_row(self, name: str, characters: int, *, note: str = "") -> None:
        row = _DocumentRow(name, characters, note=note)
        row.removed.connect(self._remove_document)
        self._documents_layout.addWidget(row)

    def _remove_document(self, row: _DocumentRow) -> None:
        self._documents = [
            (name, text) for name, text in self._documents if name != row.filename
        ]
        row.setParent(None)
        row.deleteLater()
        self._set_status(f"{row.filename} removed. Save to apply.")
        self._refresh_size()

    # ---------- load / save ----------

    def _load_existing(self) -> None:
        self._set_status("loading saved context...")
        self._run(
            "load",
            self._on_loaded,
            url=f"{self._api_url}/session/{urllib.parse.quote(self._session_id)}",
            headers=self._headers,
        )

    def _on_loaded(self, ok: bool, payload) -> None:
        if not ok:
            # Not fatal: the dialog is still usable, and saving will tell
            # the user soon enough whether the backend is reachable.
            self._set_status(f"could not load saved context: {payload}", ok=False)
            self._add_card()
            return

        context = (payload or {}).get("context", {}) or {}
        projects = context.get("projects") or []
        for index, project in enumerate(projects):
            self._add_card(project, expanded=len(projects) == 1)
        if not projects:
            self._add_card()

        manual, documents = split_documents(context.get("experience_notes", ""))
        self._notes_edit.setPlainText(manual)
        self._documents = documents
        for name, text in documents:
            self._add_document_row(name, len(text))

        self._stories_edit.setPlainText(context.get("interview_stories", "") or "")
        self._set_status("")
        self._refresh_size()

    def _payload(self) -> dict:
        projects = [card.values() for card in self._cards if not card.is_empty()]
        return {
            "session_id": self._session_id,
            "projects": projects,
            "experience_notes": compose_documents(
                self._notes_edit.toPlainText(), self._documents
            ),
            "interview_stories": self._stories_edit.toPlainText().strip(),
        }

    def _save(self) -> None:
        self._save_btn.setEnabled(False)
        self._set_status("saving...")
        self._run(
            "save",
            self._on_saved,
            url=f"{self._api_url}/session",
            headers={"Content-Type": "application/json", **self._headers},
            payload=self._payload(),
        )

    def _on_saved(self, ok: bool, payload) -> None:
        self._save_btn.setEnabled(True)
        if not ok:
            self._set_status(f"save failed: {payload}", ok=False)
            return
        summary = self._summary()
        self._set_status(f"Saved. {summary}")
        self.saved.emit(f"Project context saved -- {summary}")

    def _summary(self) -> str:
        payload = self._payload()
        projects = len(payload["projects"])
        characters = len(payload["experience_notes"]) + len(payload["interview_stories"])
        return (
            f"{projects} project{'s' if projects != 1 else ''}, "
            f"{characters:,} characters of notes"
        )

    # ---------- helpers ----------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Let in-flight requests finish before the window goes.

        A QThread destroyed while still running takes the process with it,
        and closing the dialog the moment after pressing Save is exactly
        what a user does.
        """
        for thread in list(self._threads):
            thread.wait(3000)
        super().closeEvent(event)

    def _run(self, kind: str, callback, **kwargs) -> None:
        thread = _ApiThread(kind, **kwargs)
        thread.done.connect(callback)
        thread.finished.connect(lambda: self._threads.remove(thread))
        self._threads.append(thread)  # keep a reference so it is not collected
        thread.start()

    def _set_status(self, message: str, *, ok: bool = True) -> None:
        colour = "rgba(255,255,255,150)" if ok else "#ff8a8a"
        self._status_label.setStyleSheet(f"color: {colour}; font-size: 11px;")
        self._status_label.setText(message)

    def _refresh_size(self) -> None:
        """Say how much context this is, because it is not free.

        Everything here is re-sent with every question, and providers meter
        tokens per minute -- so the honest thing is to show the size rather
        than let someone paste a 40-page handbook and wonder why answers
        started stalling. The server trims to its own budget in priority
        order regardless; this is what tells you it is about to.
        """
        payload = self._payload()
        characters = (
            sum(
                len(value)
                for project in payload["projects"]
                for value in project.values()
            )
            + len(payload["experience_notes"])
            + len(payload["interview_stories"])
        )
        self._size_label.setText(
            f"{characters:,} characters of context (~{characters // 4:,} tokens). "
            "The server sends the most relevant of this with each question, within "
            "its own budget -- project context first, then resume, then notes."
        )
