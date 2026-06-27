#!/usr/bin/env bash
#
# Set up the environment to run the tool-calling examples.
#
#   ./setup.sh
#
# Creates a virtualenv, installs dependencies, and builds shop.db.
# After it finishes, add your key to .env (or export OPENAI_API_KEY)
# and run the examples.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Creating virtual environment (.venv)"
"$PYTHON" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --quiet --upgrade pip

echo "==> Installing dependencies"
python -m pip install --quiet -r requirements.txt

echo "==> Building the database (shop.db)"
python setup_db.py

if [ ! -f .env ]; then
    echo "==> Creating .env from .env.example"
    cp .env.example .env
fi

cat <<'DONE'

Setup complete.

Next steps:
  1. Put your key in .env:        OPENAI_API_KEY=sk-...
  2. Activate the environment:    source .venv/bin/activate
  3. Load your key:               export $(grep -v '^#' .env | xargs)
  4. Run an example:              python 01_single_tool.py

DONE
