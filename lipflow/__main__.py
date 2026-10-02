"""lipflow [run] | lipflow file VIDEO | lipflow doctor"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")  # quiet MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")


def _desktop_tray() -> bool:
    return sys.platform in ("win32", "linux")


def _app():
    """Menu bar on macOS, system tray on Windows and Linux."""
    if _desktop_tray():
        from .win.app import Options, run
    else:
        from .app import Options, run
    return Options, run


def main(argv=None):
    if _desktop_tray():
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        from .win.hotkey import DEFAULT_KEY, KEYS
    else:
        from .hotkey import KEYS
        DEFAULT_KEY = "right_option"

    p = argparse.ArgumentParser(prog="lipflow", description="Silent dictation by lip reading.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="start the dictation app in the menu bar / system tray (default)")
    r.add_argument("--key", default=DEFAULT_KEY, choices=list(KEYS), help="push-to-talk key")
    r.add_argument("--beam", type=int, default=4, help="beam size (higher = slower, about the same accuracy)")
    r.add_argument("--cleanup", default="auto", choices=["auto", "claude", "local", "ollama", "basic"])
    r.add_argument("--camera", default="auto",
                   help="'auto' (the built-in camera), a camera number, part of a camera's name (Mac), "
                        "or a video file")
    r.add_argument("--copy-only", action="store_true", help="copy to the clipboard instead of pasting")
    r.add_argument("--no-preview", action="store_true", help="don't show live words while you talk")

    f = sub.add_parser("file", help="lip-read a video file")
    f.add_argument("video")
    f.add_argument("--start", type=float, default=0.0)
    f.add_argument("--end", type=float, default=None)
    f.add_argument("--beam", type=int, default=10)
    f.add_argument("--cleanup", default="auto", choices=["auto", "claude", "local", "ollama", "basic", "none"])

    sub.add_parser("doctor", help="check permissions, camera and model files")
    sub.add_parser("onboard", help="open the setup window (permissions, Wispr import, train on your face)")
    sub.add_parser("train-lm", help="fine-tune the language model on your imported phrases")
    w = sub.add_parser("import-wispr", help="learn your phrasing from your Wispr Flow history (stays local)")
    w.add_argument("--from-text", help="import a plain-text file of your writing instead (one phrase per line)")

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"run", "file", "doctor", "import-wispr", "onboard", "train-lm", "-h", "--help"}:
        argv.insert(0, "run")
    args = p.parse_args(argv)
    cmd = args.cmd

    if cmd == "file":
        from .offline import transcribe_file
        from .vsr import LipReader
        raw = transcribe_file(args.video, LipReader(beam_size=args.beam), args.start, args.end)
        print("raw:  ", raw)
        if args.cleanup != "none":
            from .cleanup import Cleaner
            c = Cleaner(args.cleanup)
            print(f"text:  {c([raw])}   [{c.describe()}]")
    elif cmd == "import-wispr":
        from .personal import PHRASES, import_wispr, save_phrases
        from .vocab import PATH as WORDS
        if args.from_text:
            stats = save_phrases(open(args.from_text, encoding="utf-8").read().splitlines()) | {"source": args.from_text}
        else:
            stats = import_wispr()
        print(f"Imported {stats['phrases']:,} phrases ({stats['words']:,} words) from {stats['source']}")
        print(f"  saved to {PHRASES}")
        if stats["new_names"]:
            print(f"  added {len(stats['new_names'])} names/terms to {WORDS}. Review them: "
                  "Lipflow menu → Edit custom words")
        print("Restart Lipflow to use them.")
    elif cmd == "train-lm":
        from .train_lm import train
        r = train()
        print(f"Your held-out phrases: perplexity {r['before']['yours']:.1f} → {r['after']['yours']:.1f}; "
              f"general text {r['before']['general']:.1f} → {r['after']['general'] or r['before']['general']:.1f}"
              f" ({'saved' if r['saved'] else 'not better, not saved'})")
    elif cmd == "onboard":
        Options, run = _app()
        run(Options(onboard=True))
    elif cmd == "doctor":
        from .doctor import doctor
        sys.exit(doctor())
    else:
        Options, run = _app()
        camera = int(args.camera) if args.camera.isdigit() else args.camera
        run(Options(key=args.key, beam=args.beam, backend=args.cleanup, camera=camera,
                    paste=not args.copy_only, live_preview=not args.no_preview))


if __name__ == "__main__":
    main()
