"""Pure text helpers: wake-phrase matching, sentence splitting, language guessing, cleanup."""
from __future__ import annotations

import difflib
import re
import unicodedata

# ---------------------------------------------------------------- normalisation

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("ł", "l")
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- wake phrase

def _ratio(a: str, b: str) -> float:
    r = difflib.SequenceMatcher(None, a, b).ratio()
    return r if abs(len(a) - len(b)) <= 1 else r * 0.8   # "claudia" is not "claude"


def split_wake(text: str, phrases: list[str], threshold: float = 0.78) -> tuple[bool, str]:
    """Checks whether `text` starts with (one of) the wake phrases, e.g. "hey claude".

    The name part must match closely, the greeting ("hey"/"hej"/"a") only loosely.
    Returns (matched, remainder) – remainder is what was said after the phrase, in its original
    spelling, so "Hey Claude, what time is it?" -> (True, "what time is it?").
    """
    original_words = text.split()
    norm = [normalize(w) for w in original_words]
    idx = [i for i, w in enumerate(norm) if w]      # map back to original words
    words = [norm[i] for i in idx]
    if not words:
        return False, ""

    targets = []
    for p in phrases:
        parts = normalize(p).split()
        if parts:
            targets.append((parts[:-1], parts[-1]))  # (greeting words, name)

    best_end = None
    best = 0.0
    for start in range(0, min(2, len(words))):     # allow one filler word before ("okay hey claude")
        for greet, name in targets:
            g = " ".join(greet)
            # "hey claude" as two words
            if greet and start + 1 < len(words):
                name_s = _ratio(words[start + 1], name)
                score = 0.25 * difflib.SequenceMatcher(None, words[start], g).ratio() + 0.75 * name_s
                if name_s >= threshold and score >= threshold and score > best:
                    best, best_end = score, start + 2
            # "heyclaude" merged into one word, or a one-word phrase
            merged = g.replace(" ", "") + name
            score = _ratio(words[start], merged)
            if score >= threshold and score > best and (greet or start == 0):
                best, best_end = score, start + 1

    if best_end is None:
        return False, ""
    if best_end >= len(idx):
        return True, ""
    remainder = " ".join(original_words[idx[best_end]:])
    return True, remainder.lstrip(" ,.!?-–—:;").strip()


# ---------------------------------------------------------------- whisper noise

_HALLUCINATIONS = {
    "dziekuje", "dziekuje za uwage", "dzieki", "thank you", "thanks for watching", "thank you for watching",
    "napisy wykonane przez", "napisy stworzone przez spolecznosc amara org", "subtitles by the amara org community",
    "you", "bye", "uh", "um", "hmm", "mhm", "",
}


def is_noise(text: str) -> bool:
    n = normalize(text)
    if n in _HALLUCINATIONS:
        return True
    if n.startswith("napisy") or "amara org" in n:
        return True
    return len(n) < 2


# ---------------------------------------------------------------- streaming sentences

_SENT_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
_ABBREV = re.compile(r"(\b(?:np|tzn|itd|itp|dr|mgr|inż|ul|godz|min|ok|tj|m\.in|e\.g|i\.e|mr|mrs|ms|vs|st|prof)\.|\d\.)$", re.I)


class SentenceSplitter:
    """Feed streamed text chunks, get complete sentences out as soon as they end."""

    def __init__(self, min_chars: int = 12):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, chunk: str) -> list[str]:
        self.buf += chunk
        out = []
        start = 0
        for m in _SENT_END.finditer(self.buf):
            candidate = self.buf[start:m.start()].strip()
            if _ABBREV.search(candidate) or len(candidate) < self.min_chars and m.group(0).strip() == "":
                continue  # "np." or a very short fragment like "Tak." – wait for more
            if candidate:
                out.append(candidate)
            start = m.end()
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


# ---------------------------------------------------------------- spoken cleanup

def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)           # code blocks – unreadable aloud
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # [link](url) -> link
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>|~]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)  # emoji
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- language guess

_PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_PL_WORDS = {"i", "w", "na", "nie", "to", "jest", "się", "że", "z", "do", "jak", "co", "tak", "ale", "o", "dla",
             "czy", "już", "jestem", "masz", "możesz", "jutro", "dzisiaj", "teraz", "będzie", "od", "po", "przez"}
_EN_WORDS = {"the", "is", "are", "and", "to", "of", "a", "in", "it", "you", "i", "that", "for", "on", "with",
             "this", "what", "your", "can", "will", "be", "it's", "i'm", "there", "have", "at"}


def guess_lang(text: str, default: str) -> str:
    if any(c in _PL_CHARS for c in text):
        return "pl"
    words = re.findall(r"[a-ząćęłńóśźż']+", text.lower())
    pl = sum(w in _PL_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if en >= 2 and en > pl * 2:
        return "en"
    if pl >= 2 and pl > en * 2:
        return "pl"
    return default
