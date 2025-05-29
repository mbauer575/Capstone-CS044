#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="venv2"

# 1) Create & bootstrap venv only once
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR?"
  python3 -m venv "$VENV_DIR" --system-site-packages

  echo "Activating venv and installing requirements?"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  pip install --upgrade pip
  pip install -r requirements.txt
else
  # Just activate on subsequent runs
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

# 2) If PIL.ImageTk still won?t import, force-reinstall Pillow inside the venv
if ! python -c "from PIL import ImageTk" &>/dev/null; then
  echo "ImageTk missing?rebuilding Pillow in venv?"
  pip install --upgrade \
    --force-reinstall \
    --ignore-installed \
    --no-binary :all: \
    Pillow
fi

echo "Launching UI?"
python UI.py

# optional ?press any key to exit?
read -n1 -r -p "Press any key to exit?" key

pause
