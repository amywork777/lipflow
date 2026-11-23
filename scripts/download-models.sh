#!/usr/bin/env bash
# Model weights (~1.2 GB). Pass --samples for the public-domain clips used by tests.
set -euo pipefail
cd "$(dirname "$0")/.."

get() {  # url dest
  [ -s "$2" ] && { echo "✓ $2"; return; }
  mkdir -p "$(dirname "$2")"
  echo "↓ $2"
  # Wikimedia answers 429 to curl's default User-Agent; name the client and retry rate limits.
  curl -fL --retry 4 --retry-delay 5 -A "lipflow-setup/1.0 (https://github.com/amywork777/lipflow)" \
    --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

HF=https://huggingface.co
# Auto-AVSR visual-only model trained on LRS3 (WER 19.1%) + subword RNN language model
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json models/vsr/model.json
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth  models/vsr/model.pth
get $HF/Amanvir/lm_en_subword/resolve/main/model.json  models/lm/model.json
get $HF/Amanvir/lm_en_subword/resolve/main/model.pth   models/lm/model.pth
# SentencePiece tokenizer for the LM (needed to train on your phrases and your face)
get https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model models/lm/unigram5000.model
# MediaPipe face landmarker
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    models/face_landmarker.task

if [[ "${1:-}" == "--samples" ]]; then
  # Public-domain White House weekly addresses (Wikimedia Commons), used by tests/test_pipeline.py
  C=https://upload.wikimedia.org/wikipedia/commons/transcoded
  get "$C/c/ce/2016-03-12_President_Obama%27s_Weekly_Address.webm/2016-03-12_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" samples/2016-03-12.mov
  get "$C/2/29/2017-01-07_President_Obama%27s_Weekly_Address.webm/2017-01-07_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" samples/2017-01-07.mov
fi
