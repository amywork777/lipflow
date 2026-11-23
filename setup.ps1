# Lipflow for Windows: install dependencies, download the models (~1.2 GB) and add a Start menu shortcut.
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-Samples] [-NoShortcut]
param([switch]$Samples, [switch]$NoShortcut)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is 10x slower with the progress bar
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

function Get-File($url, $dest) {
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) { Write-Host "ok  $dest"; return }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    Write-Host "get $dest"
    # Wikimedia answers 429 to curl's default User-Agent; name the client and retry rate limits.
    curl.exe -fL --retry 4 --retry-delay 5 -A "lipflow-setup/1.0 (https://github.com/amywork777/lipflow)" --progress-bar -o "$dest.part" $url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    Move-Item -Force "$dest.part" $dest
}

$HF = "https://huggingface.co"
# Auto-AVSR visual-only model trained on LRS3 (WER 19.1%) + subword RNN language model
Get-File "$HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json" "models/vsr/model.json"
Get-File "$HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth"  "models/vsr/model.pth"
Get-File "$HF/Amanvir/lm_en_subword/resolve/main/model.json"  "models/lm/model.json"
Get-File "$HF/Amanvir/lm_en_subword/resolve/main/model.pth"   "models/lm/model.pth"
# SentencePiece tokenizer for the LM (needed to train on your phrases and your face)
Get-File "https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model" "models/lm/unigram5000.model"
# MediaPipe face landmarker
Get-File "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
    "models/face_landmarker.task"

if ($Samples) {
    # Public-domain White House weekly addresses (Wikimedia Commons), used by tests/test_pipeline.py
    $C = "https://upload.wikimedia.org/wikipedia/commons/transcoded"
    Get-File "$C/c/ce/2016-03-12_President_Obama%27s_Weekly_Address.webm/2016-03-12_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" "samples/2016-03-12.mov"
    Get-File "$C/2/29/2017-01-07_President_Obama%27s_Weekly_Address.webm/2017-01-07_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" "samples/2017-01-07.mov"
}

if (-not $NoShortcut) {
    # Start menu entry that runs the tray app without a console window
    $data = Join-Path $env:APPDATA "Lipflow"
    New-Item -ItemType Directory -Force -Path $data | Out-Null
    $ico = Join-Path $data "Lipflow.ico"
    uv run python -c "import sys; from lipflow.win.hud import tray_image; tray_image(size=256).save(sys.argv[1], sizes=[(16,16),(32,32),(48,48),(256,256)])" $ico
    $lnk = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Lipflow.lnk"
    $sh = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $sh.TargetPath = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
    $sh.Arguments = "-X utf8 -m lipflow"
    $sh.WorkingDirectory = $PSScriptRoot
    $sh.IconLocation = $ico
    $sh.Description = "Silent dictation by lip reading"
    $sh.Save()
    Write-Host "ok  Start menu shortcut: $lnk"
}

Write-Host ""
Write-Host "Done. Open Lipflow from the Start menu (it lives in the system tray, next to the clock)."
Write-Host "The first launch walks you through your Wispr Flow words and ~24 practice sentences."
Write-Host "Check everything with:  uv run lipflow doctor"
