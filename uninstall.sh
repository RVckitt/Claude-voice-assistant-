#!/usr/bin/env bash
# Removes Vox. Your config (~/.config/vox) is kept unless you pass --purge.
set -euo pipefail
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
CONF="${XDG_CONFIG_HOME:-$HOME/.config}"
"$HOME/.local/bin/vox" ctl quit 2>/dev/null || true
rm -rf "$DATA/vox/venv"
rm -f "$HOME/.local/bin/vox" "$DATA/applications/vox.desktop" "$CONF/autostart/vox.desktop" \
      "$DATA/icons/hicolor/scalable/apps/vox.svg"
if [[ "${1:-}" == "--purge" ]]; then
    rm -rf "$DATA/vox" "$CONF/vox" "${XDG_STATE_HOME:-$HOME/.local/state}/vox"
    echo "Removed Vox, its voices and config. (Whisper models stay in ~/.cache/huggingface.)"
else
    echo "Removed Vox. Config and voices kept – run with --purge to delete them too."
fi
