"""Talks to Claude: short conversation memory, voice-friendly system prompt, streamed replies."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Iterator
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

LANG_NAMES = {"pl": "Polish", "en": "English"}

SYSTEM_PROMPT = """You are Claude, running as a hands-free voice assistant on {user}'s Linux laptop \
(an app called Vox). Everything you write is converted to speech and played through the speakers.

Rules for spoken answers:
- Be brief: usually 1-3 sentences. Give the answer first; offer more detail only if it's useful.
- Plain sentences only: no markdown, bullet points, headings, tables, emoji, URLs or code blocks.
- Write numbers, units and symbols the way they should be read aloud when that helps.
- Reply in {lang} (the language the user just spoke) unless they ask for another language.
- The speech-to-text can mishear words. If a request looks garbled, make your best guess, \
or ask a short clarifying question.
- You cannot see the screen or control the computer; say so plainly if asked.

Current date and time: {now} ({tz}).{location}{extra}"""


class Brain:
    def __init__(self, api_key: str, model: str, max_tokens: int = 600, web_search: bool = False,
                 memory_minutes: float = 5, user_name: str = "the user", location: str = "",
                 timezone: str = "Europe/Warsaw", extra_instructions: str = "", _client=None):
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic(api_key=api_key or None)
        self.client = _client
        self.model = model
        self.max_tokens = max_tokens
        self.web_search = web_search
        self.memory_s = memory_minutes * 60
        self.user = user_name or "the user"
        self.location = location
        self.tz = timezone
        self.extra = extra_instructions
        self.history: list[dict] = []
        self._last = 0.0

    def reset(self):
        self.history.clear()

    def _system(self, lang: str) -> str:
        try:
            now = datetime.now(ZoneInfo(self.tz))
        except Exception:  # noqa: BLE001
            now = datetime.now()
        return SYSTEM_PROMPT.format(
            user=self.user, lang=LANG_NAMES.get(lang, lang),
            now=now.strftime("%A, %d %B %Y, %H:%M"), tz=self.tz,
            location=f"\nUser's location: {self.location}." if self.location else "",
            extra=f"\n\n{self.extra}" if self.extra else "",
        )

    def _tools(self) -> list[dict]:
        if not self.web_search:
            return []
        tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
        if self.location:
            city = self.location.split(",")[0].strip()
            tool["user_location"] = {"type": "approximate", "city": city, "timezone": self.tz}
        return [tool]

    def ask(self, text: str, lang: str, cancelled=lambda: False) -> Iterator[str]:
        """Yields reply text as it streams in. Conversation memory is kept between calls."""
        if time.monotonic() - self._last > self.memory_s:
            self.reset()
        self.history.append({"role": "user", "content": text})
        reply = ""
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=self._system(lang),
                      messages=self.history)
        tools = self._tools()
        if tools:
            kwargs["tools"] = tools
        try:
            with self.client.messages.stream(**kwargs) as stream:
                for chunk in stream.text_stream:
                    if cancelled():
                        break
                    reply += chunk
                    yield chunk
        except Exception:
            self.history.pop()  # don't leave a dangling user turn behind
            raise
        finally:
            self._last = time.monotonic()
        if reply.strip():
            self.history.append({"role": "assistant", "content": reply.strip()})
        else:
            self.history.pop()
        # keep the context small – voice chats don't need a long memory
        if len(self.history) > 20:
            self.history = self.history[-20:]
