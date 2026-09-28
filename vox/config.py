"""Configuration: ~/.config/vox/config.toml (created on first run)."""
from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

APP = "vox"


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default)


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / APP
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share") / APP
STATE_DIR = _xdg("XDG_STATE_HOME", ".local/state") / APP
CONFIG_PATH = CONFIG_DIR / "config.toml"
VOICES_DIR = DATA_DIR / "voices"
LOG_PATH = STATE_DIR / "vox.log"


def runtime_socket() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/{APP}-{os.getuid()}"
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(base) / f"{APP}.sock"


DEFAULT_TOML = """\
# Vox — voice assistant powered by Claude.
# Edit this file, then choose "Restart" from the tray menu (or restart vox).

[wake]
# "whisper"      – works out of the box with any phrase (transcribes short speech bursts locally)
# "openwakeword" – lighter on CPU, needs a trained model file (see README)
engine = "whisper"
phrase = "hey claude"
# Other ways speech-to-text may hear the phrase (Polish accent, mishearings):
aliases = ["hej claude", "hey cloud", "hej klaud", "hey klaud", "hej klod", "hey clod", "hey claud", "ej claude", "hej cloud", "hey clawed"]
whisper_model = "base"       # small model used only to spot the wake phrase
sensitivity = 0.78           # 0..1 – lower = triggers more easily (and falsely)
openwakeword_model = ""      # path to .onnx model, e.g. ~/.local/share/vox/hey_claude.onnx
openwakeword_threshold = 0.5

[audio]
input_device = ""            # "" = system default; or a name fragment / index (vox --list-devices)
output_device = ""
vad_aggressiveness = 2       # 0..3 – higher = ignores more background noise
min_volume = 60              # ignore sounds quieter than this (raise if fans/hiss trigger it, lower for a quiet mic)
silence_ms = 900             # how long you must be quiet for the utterance to end
max_utterance_s = 25
command_timeout_s = 6        # after the beep, how long to wait for you to start talking
follow_up_s = 5              # after an answer, listen this long for a follow-up without the wake word (0 = off)
beep = true

[stt]
model = "small"              # tiny / base / small / medium / large-v3 / turbo – bigger = better Polish, slower
device = "auto"              # auto / cpu / cuda
compute_type = "auto"
languages = ["pl", "en"]     # auto-detect, but only between these

[claude]
api_key = ""                 # leave empty to use the ANTHROPIC_API_KEY environment variable
model = "claude-haiku-4-5-20251001"   # fastest; "claude-sonnet-5" is smarter but slower
max_tokens = 600
web_search = false           # let Claude search the web (weather, news…) – costs extra per search
memory_minutes = 5           # forget the conversation after this many idle minutes
user_name = "Szymon"
location = ""                # e.g. "Poznań, Poland" – helps with weather / local questions
timezone = "Europe/Warsaw"
extra_instructions = ""

[tts]
speed = 1.0                  # >1 faster, <1 slower

[tts.voices]                 # Piper voice per language (https://rhasspy.github.io/piper-samples/)
pl = "pl_PL-gosia-medium"
en = "en_US-lessac-medium"

[ui]
popup = "auto"               # auto / bubble / notify / off
"""


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    def get(self, section: str, key: str, default=None):
        return self.raw.get(section, {}).get(key, default)

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})

    @property
    def api_key(self) -> str:
        return self.get("claude", "api_key") or os.environ.get("ANTHROPIC_API_KEY", "")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_config(path: Path = CONFIG_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_TOML, encoding="utf-8")
        path.chmod(0o600)  # may contain the API key
    return path


def load_config(path: Path | None = None) -> Config:
    defaults = tomllib.loads(DEFAULT_TOML)
    path = path or ensure_config()
    user = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return Config(_merge(defaults, user))


def set_api_key(key: str, path: Path = CONFIG_PATH) -> None:
    """Write the API key into the config file, keeping the rest of the file intact."""
    ensure_config(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    in_claude = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("["):
            in_claude = s == "[claude]"
        elif in_claude and s.startswith("api_key"):
            comment = line[line.index("#"):] if "#" in line else ""
            lines[i] = f'api_key = "{key}"' + (f"                 {comment}" if comment else "")
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
