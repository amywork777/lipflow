"""Where each person's data lives. Shared model weights stay in the repo's models/; everything
learned from *you* (clips, phrases, personal models, settings) goes here.

Override with LIPFLOW_HOME (tests use a temp dir so they never touch your real data)."""
import os
import sys

WINDOWS = sys.platform == "win32"

if WINDOWS:  # %APPDATA%\Lipflow
    _DEFAULT_HOME = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Lipflow")
else:
    _DEFAULT_HOME = os.path.expanduser("~/Library/Application Support/Lipflow")
HOME = os.environ.get("LIPFLOW_HOME") or _DEFAULT_HOME
PERSONAL_MODELS = os.path.join(HOME, "models")
PERSONAL_VSR = os.path.join(PERSONAL_MODELS, "vsr_face.pth")
PERSONAL_LM = os.path.join(PERSONAL_MODELS, "lm_phrasing.pth")

# The app bundle's launcher sets LIPFLOW_APP=1: permissions then belong to "Lipflow", not the terminal.
# On Windows nothing is granted per app, so the name only shows up in messages.
WHO = "Lipflow" if os.environ.get("LIPFLOW_APP") or WINDOWS else "your terminal"


def personal_vsr(language="en"):
    return PERSONAL_VSR if language == "en" else os.path.join(PERSONAL_MODELS, language, "vsr_face.pth")
