#!/usr/bin/env bash
# Same as repo-root scripts/download-models.sh (for upstream PR layout).
set -euo pipefail
cd "$(dirname "$0")/.."

get() {
  [ -s "$2" ] && { echo "✓ $2"; return; }
  mkdir -p "$(dirname "$2")"
  echo "↓ $2"
  curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

HF=https://huggingface.co
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json models/vsr/model.json
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth models/vsr/model.pth
get $HF/Amanvir/lm_en_subword/resolve/main/model.json models/lm/model.json
get $HF/Amanvir/lm_en_subword/resolve/main/model.pth models/lm/model.pth
get https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model models/lm/unigram5000.model
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  models/face_landmarker.task

echo "Done. Run: uv run lipflow doctor"
