"""System-tray UI (PySide6). Works with KDE Plasma's tray on X11 and Wayland."""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import QObject, QPoint, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QInputDialog, QLabel, QMenu, QSystemTrayIcon, QVBoxLayout, QWidget)

from . import engine as E
from .config import CONFIG_PATH, LOG_PATH

log = logging.getLogger(__name__)

COLORS = {
    E.LOADING: "#8A8A8A",
    E.IDLE: "#111111",
    E.LISTENING: "#2F6BFF",
    E.THINKING: "#F2A20C",
    E.SPEAKING: "#12B76A",
    E.MUTED: "#8A8A8A",
    E.ERROR: "#E5484D",
}
LABELS = {
    E.LOADING: "Loading", E.IDLE: "Ready", E.LISTENING: "Listening", E.THINKING: "Thinking",
    E.SPEAKING: "Speaking", E.MUTED: "Muted", E.ERROR: "Error",
}


def make_icon(state: str, size: int = 64) -> QIcon:
    """Minimal round icon: white ring + filled dot, colour = state. Muted gets a slash."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    color = QColor(COLORS.get(state, "#111111"))
    m = size * 0.06
    p.setPen(QPen(QColor("#FFFFFF"), size * 0.07))
    p.setBrush(color)
    p.drawEllipse(QRectF(m, m, size - 2 * m, size - 2 * m))
    inner = size * (0.20 if state in (E.IDLE, E.LOADING, E.MUTED) else 0.30)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#FFFFFF"))
    c = size / 2
    p.drawEllipse(QRectF(c - inner / 2, c - inner / 2, inner, inner))
    if state == E.MUTED:
        p.setPen(QPen(QColor("#FFFFFF"), size * 0.08, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPoint(int(size * 0.25), int(size * 0.75)), QPoint(int(size * 0.75), int(size * 0.25)))
    p.end()
    return QIcon(pm)


class QtEvents(QObject):
    """Engine callbacks -> Qt signals (safe to emit from the engine thread)."""
    state_sig = Signal(str, str)
    user_sig = Signal(str)
    reply_sig = Signal(str, bool)
    error_sig = Signal(str)

    def state(self, state, detail=""):
        self.state_sig.emit(state, detail)

    def user(self, text):
        self.user_sig.emit(text)

    def reply(self, text, done):
        self.reply_sig.emit(text, done)

    def error(self, message):
        self.error_sig.emit(message)


class Bubble(QWidget):
    """Small floating card in the bottom-right corner showing what was heard and the answer."""

    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedWidth(380)
        self.card = QWidget(self)
        self.card.setObjectName("card")
        self.card.setStyleSheet("""
            #card { background: rgba(17,17,17,235); border-radius: 14px; }
            QLabel { color: #F4F4F4; background: transparent; }
            #status { color: #9A9A9A; font-size: 11px; letter-spacing: 1px; }
            #you { color: #BDBDBD; }
        """)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(6)
        self.status = QLabel(objectName="status")
        self.you = QLabel(objectName="you")
        self.answer = QLabel()
        f = QFont()
        f.setPointSize(11)
        for w in (self.you, self.answer):
            w.setWordWrap(True)
            w.setFont(f)
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        lay.addWidget(self.you)
        lay.addWidget(self.answer)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.card)
        self.hide_timer = QTimer(self, singleShot=True, timeout=self.hide)

    def update_content(self, state: str, detail: str, you: str, answer: str):
        label = LABELS.get(state, state)
        if detail.lower().startswith(label.lower()):
            detail = ""
        self.status.setText(f"● {label.upper()}   {detail}".strip())
        self.status.setStyleSheet(f"color: {COLORS.get(state, '#9A9A9A')}; font-size: 11px;")
        self.you.setText(f"„{you}”" if you else "")
        self.you.setVisible(bool(you))
        shown = answer if len(answer) < 600 else "…" + answer[-600:]
        self.answer.setText(shown)
        self.answer.setVisible(bool(answer))
        self.adjustSize()
        self._place()
        self.hide_timer.stop()
        if not self.isVisible():
            self.show()

    def linger(self, ms: int = 7000):
        self.hide_timer.start(ms)

    def _place(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.move(g.right() - self.width() - 16, g.bottom() - self.height() - 16)


class TrayApp:
    def __init__(self, app: QApplication, engine_factory, popup: str = "auto"):
        self.app = app
        self.events = QtEvents()
        self.engine = engine_factory(self.events)
        self.state = E.LOADING
        self.detail = ""
        self.you = ""
        self.answer = ""
        self.icons = {s: make_icon(s) for s in COLORS}

        if popup == "auto":
            popup = "notify" if QGuiApplication.platformName() == "wayland" else "bubble"
        self.popup = popup
        self.bubble = Bubble() if popup == "bubble" else None

        self.tray = QSystemTrayIcon(self.icons[E.LOADING])
        self.tray.setToolTip("Vox – loading")
        self.tray.activated.connect(self._on_activated)
        self.menu = QMenu()
        self.status_action = self.menu.addAction("Loading…")
        self.status_action.setEnabled(False)
        self.menu.addSeparator()
        self.menu.addAction("Listen now", self.engine.listen_now)
        self.menu.addAction("Ask by typing…", self._ask_typed)
        self.menu.addAction("Stop speaking", self.engine.stop)
        self.mute_action = QAction("Mute microphone", self.menu, checkable=True)
        self.mute_action.toggled.connect(self.engine.set_muted)
        self.menu.addAction(self.mute_action)
        self.menu.addAction("New conversation", self.engine.new_conversation)
        self.menu.addSeparator()
        self.menu.addAction("Open settings", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(CONFIG_PATH))))
        self.menu.addAction("Open log", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_PATH))))
        self.menu.addAction("Restart", self.restart)
        self.menu.addAction("Quit", self.quit)
        self.tray.setContextMenu(self.menu)
        self.tray.show()

        self.events.state_sig.connect(self._on_state)
        self.events.user_sig.connect(self._on_user)
        self.events.reply_sig.connect(self._on_reply)
        self.events.error_sig.connect(self._on_error)

    # ---------------------------------------------------------------- slots
    def _on_state(self, state: str, detail: str):
        self.state, self.detail = state, detail
        self.tray.setIcon(self.icons.get(state, self.icons[E.IDLE]))
        self.tray.setToolTip(f"Vox – {LABELS.get(state, state)}" + (f": {detail}" if detail else ""))
        self.status_action.setText(f"{LABELS.get(state, state)} — {detail}" if detail else LABELS.get(state, state))
        if self.mute_action.isChecked() != (state == E.MUTED) and state in (E.MUTED, E.IDLE):
            self.mute_action.blockSignals(True)
            self.mute_action.setChecked(state == E.MUTED)
            self.mute_action.blockSignals(False)
        if self.bubble:
            if state in (E.LISTENING, E.THINKING, E.SPEAKING):
                if state == E.LISTENING and self.detail == "Listening…":
                    self.you, self.answer = "", ""
                self.bubble.update_content(state, detail, self.you, self.answer)
            elif state in (E.IDLE, E.MUTED) and self.bubble.isVisible():
                self.bubble.update_content(state, "", self.you, self.answer)
                self.bubble.linger()

    def _on_user(self, text: str):
        self.you, self.answer = text, ""
        if self.bubble:
            self.bubble.update_content(self.state, self.detail, self.you, "")

    def _on_reply(self, text: str, done: bool):
        self.answer = text
        if self.bubble and (self.bubble.isVisible() or text):
            self.bubble.update_content(self.state, self.detail, self.you, text)
        if done and text and self.popup == "notify":
            self.tray.showMessage(self.you[:80] or "Claude", text, self.icons[E.SPEAKING], 12000)

    def _on_error(self, message: str):
        log.error(message)
        if self.popup != "off":
            self.tray.showMessage("Vox", message, QSystemTrayIcon.Warning, 10000)

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:        # left click = push to talk
            if self.state == E.SPEAKING:
                self.engine.stop()
            else:
                self.engine.listen_now()
        elif reason == QSystemTrayIcon.MiddleClick:
            self.mute_action.toggle()

    def _ask_typed(self):
        text, ok = QInputDialog.getText(None, "Ask Claude", "Question:")
        if ok and text.strip():
            self.engine.ask(text.strip())

    # ---------------------------------------------------------------- lifecycle
    def quit(self):
        self.engine.shutdown()
        self.tray.hide()
        self.app.quit()

    def restart(self):
        self.engine.shutdown()
        self.tray.hide()
        os.execv(sys.executable, [sys.executable, "-m", "vox", *sys.argv[1:]])
