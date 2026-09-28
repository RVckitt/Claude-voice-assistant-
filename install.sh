#!/usr/bin/env bash
# Vox installer for CachyOS / Arch (also works on Debian/Ubuntu/Fedora).
#   ./install.sh                 install + autostart on login
#   ./install.sh --no-autostart  install without autostart
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
CONF="${XDG_CONFIG_HOME:-$HOME/.config}"
PREFIX="$DATA/vox"
BIN="$HOME/.local/bin"
AUTOSTART=1
[[ "${1:-}" == "--no-autostart" ]] && AUTOSTART=0

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "System packages (PortAudio for the microphone, uv for Python)"
if command -v pacman >/dev/null; then
    sudo pacman -S --needed --noconfirm portaudio uv
elif command -v apt-get >/dev/null; then
    sudo apt-get install -y libportaudio2 curl
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
elif command -v dnf >/dev/null; then
    sudo dnf install -y portaudio uv
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

say "Python environment in $PREFIX/venv (Python 3.12, isolated from the system)"
uv venv --allow-existing --python 3.12 "$PREFIX/venv"
uv pip install --python "$PREFIX/venv/bin/python" --reinstall-package vox-assistant "$SRC"

say "Launcher, icon and menu entry"
mkdir -p "$BIN" "$DATA/applications" "$DATA/icons/hicolor/scalable/apps"
ln -sf "$PREFIX/venv/bin/vox" "$BIN/vox"
install -m644 "$SRC/packaging/vox.svg" "$DATA/icons/hicolor/scalable/apps/vox.svg"
sed "s|@VOX@|$BIN/vox|" "$SRC/packaging/vox.desktop" > "$DATA/applications/vox.desktop"
if [[ $AUTOSTART == 1 ]]; then
    mkdir -p "$CONF/autostart"
    cp "$DATA/applications/vox.desktop" "$CONF/autostart/vox.desktop"
    echo "Vox will start automatically when you log in (remove $CONF/autostart/vox.desktop to disable)."
fi

say "Setup: API key + downloading speech models (~600 MB the first time)"
"$BIN/vox" --setup

case ":$PATH:" in *":$BIN:"*) ;; *)
    echo
    echo "NOTE: $BIN is not on your PATH. Add it, e.g. for fish:  fish_add_path ~/.local/bin"
    echo "      for bash/zsh:  echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc" ;;
esac

say "Done! Start it now with:  vox     (or from the app menu: Vox)"
echo "Tip: bind  vox ctl listen  to a keyboard shortcut in System Settings → Shortcuts for push-to-talk."
