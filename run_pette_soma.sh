#!/usr/bin/env bash
# Run Pette with SOMA watching the screen and suggesting actions.
#
#   ./run_pette_soma.sh                 # set up both venvs if needed, start SOMA, then Pette
#   ./run_pette_soma.sh --demo          # extra flags go to Pette (main.py): --demo, --always-show, --no-camera ...
#   ./run_pette_soma.sh --diagnose      # one SOMA step on the current screen: why it would speak or stay silent
#
# Running this script is your opt-in for SOMA to capture the screen (consent is stored in
# SOMA/data/consent.json; revoke it there or with SOMA's desktop UI). Screenshots stay in memory, are
# never written to disk, and password managers / private windows / sign-in prompts are skipped.
# Suggestions appear only while Pette is on screen (VS Code and terminals; --always-show for every app)
# and nothing runs until you click "Do it".
#
# Env: SOMA_CONFIG (default configs/pette.yaml), SOMA_CHECKPOINT, SOMA_INTERVAL (s, default 2),
#      SOMA_SYNTHETIC=1 (synthetic screens instead of yours), SKIP_SETUP=1 (don't touch the venvs).
# Hotkeys: Ctrl+Shift+Space talk, Ctrl+Alt+S suggestions on/off, Ctrl+Alt+C camera, Ctrl+Alt+W world panel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOMA_DIR="$ROOT/SOMA"
SOMA_CONFIG="${SOMA_CONFIG:-configs/pette.yaml}"
SOMA_INTERVAL="${SOMA_INTERVAL:-2}"
PY_VERSION="3.12"

say() { printf '\033[1;35m[pette+soma]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[pette+soma]\033[0m %s\n' "$*" >&2; exit 1; }

# Native Windows paths for Python when running under Git Bash / MSYS.
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi; }

venv_python() {   # $1 = venv dir
    if [[ -x "$1/Scripts/python.exe" ]]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi
}

stamp_ok() {      # $1 = stamp file, $2... = files whose content decides if setup must re-run
    [[ -f "$1" ]] && [[ "$(cat "$1")" == "$(cat "${@:2}" | cksum)" ]]
}

write_stamp() { cat "${@:2}" | cksum > "$1"; }

setup() {
    command -v uv >/dev/null 2>&1 || die "uv is required (https://docs.astral.sh/uv/): pip install uv"

    # Pette: its own venv (mediapipe needs opencv-contrib; SOMA's RapidOCR pulls opencv-python).
    if [[ ! -d "$ROOT/.venv" ]]; then
        say "Creating Pette's venv (.venv)"
        uv venv "$ROOT/.venv" --python "$PY_VERSION" -q
    fi
    if ! stamp_ok "$ROOT/.venv/.pette-setup" "$ROOT/requirements.txt"; then
        say "Installing Pette's requirements"
        uv pip install --python "$(venv_python "$ROOT/.venv")" -q -r "$ROOT/requirements.txt"
        write_stamp "$ROOT/.venv/.pette-setup" "$ROOT/requirements.txt"
    fi
    if [[ ! -f "$ROOT/models/face_landmarker.task" || ! -f "$ROOT/models/hand_landmarker.task" ]]; then
        say "Downloading MediaPipe face/hand models for the camera orb"
        (cd "$ROOT" && "$(venv_python "$ROOT/.venv")" tools/get_models.py) || say "Model download failed; camera features stay off"
    fi

    # SOMA: CPU PyTorch + capture / OCR extras, in SOMA/.venv.
    if [[ ! -d "$SOMA_DIR/.venv" ]]; then
        say "Creating SOMA's venv (SOMA/.venv)"
        uv venv "$SOMA_DIR/.venv" --python "$PY_VERSION" -q
    fi
    if ! stamp_ok "$SOMA_DIR/.venv/.soma-setup" "$SOMA_DIR/pyproject.toml"; then
        local spy; spy="$(venv_python "$SOMA_DIR/.venv")"
        say "Installing SOMA (PyTorch CPU, mss, RapidOCR); the first time takes a few minutes"
        uv pip install --python "$spy" -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
        (cd "$SOMA_DIR" && uv pip install --python "$spy" -q -e ".[capture,ocr]")
        write_stamp "$SOMA_DIR/.venv/.soma-setup" "$SOMA_DIR/pyproject.toml"
    fi
}

soma_args=(--config "$SOMA_CONFIG" --interval "$SOMA_INTERVAL" --grant-screen)
[[ -n "${SOMA_CHECKPOINT:-}" ]] && soma_args+=(--checkpoint "$SOMA_CHECKPOINT")
[[ "${SOMA_SYNTHETIC:-0}" == "1" ]] && soma_args+=(--synthetic)

[[ "${SKIP_SETUP:-0}" == "1" ]] || setup
PETTE_PY="$(venv_python "$ROOT/.venv")"
SOMA_PY="$(venv_python "$SOMA_DIR/.venv")"
[[ -x "$PETTE_PY" && -x "$SOMA_PY" ]] || die "venvs missing; run without SKIP_SETUP=1"

if [[ "${1:-}" == "--diagnose" ]]; then
    say "One SOMA step on your current screen (switch to the window you want checked within 3 s)"
    sleep 3
    cd "$SOMA_DIR" && PYTHONIOENCODING=utf-8 exec "$SOMA_PY" -m scripts.serve "${soma_args[@]}" --once
fi

if [[ ! -f "$ROOT/.env" && -z "${GROQ_API_KEY:-}" && " $* " != *" --demo "* ]]; then
    say "No .env / GROQ_API_KEY: SOMA suggestions will show, but drafting needs Groq (or pass --demo)."
fi

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pette-soma.XXXXXX")"
ENDPOINT="$RUN_DIR/endpoint.json"
SOMA_LOG="$RUN_DIR/soma.log"
SOMA_PID=""

cleanup() {
    if [[ -n "$SOMA_PID" ]] && kill -0 "$SOMA_PID" 2>/dev/null; then
        kill "$SOMA_PID" 2>/dev/null || true
        wait "$SOMA_PID" 2>/dev/null || true
    fi
    rm -rf "$RUN_DIR"   # endpoint (URL + token) and log
}
trap cleanup EXIT INT TERM

say "Starting SOMA ($SOMA_CONFIG) - it will capture your screen with consent"
(cd "$SOMA_DIR" && PYTHONIOENCODING=utf-8 exec "$SOMA_PY" -m scripts.serve "${soma_args[@]}" \
    --endpoint-file "$(winpath "$ENDPOINT")") >"$SOMA_LOG" 2>&1 &
SOMA_PID=$!

for _ in $(seq 1 240); do
    [[ -s "$ENDPOINT" ]] && break
    kill -0 "$SOMA_PID" 2>/dev/null || { cat "$SOMA_LOG" >&2; die "SOMA exited during startup (log above)"; }
    sleep 0.5
done
[[ -s "$ENDPOINT" ]] || { cat "$SOMA_LOG" >&2; die "SOMA did not start within 2 minutes"; }
say "$(tail -n 1 "$SOMA_LOG")"

say "Starting Pette (extra flags: ${*:-none}). Quit from the tray; SOMA stops with her."
cd "$ROOT"
"$PETTE_PY" main.py --soma "$(winpath "$ENDPOINT")" "$@"
