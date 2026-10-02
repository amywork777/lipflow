"""First-run setup on Windows: welcome → import Wispr Flow → mouth ~24 sentences → train on your face.

Same steps as the macOS setup window (minus permissions: Windows only needs the camera switch in
Settings → Privacy & security → Camera). Practice uses the normal push-to-talk key.
"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk

from ..practice import N_SENTENCES, practice_sentences, save_clip, saved_clips
from .hud import ACCENT, AMBER, BG, DIM, FG, FONT, GREEN, photo

WW, WH = 600, 560


class Setup:
    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Lipflow setup")
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        sw, sh = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
        self.win.geometry(f"{WW}x{WH}+{(sw - WW) // 2}+{(sh - WH) // 3}")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.page = None
        self.video = None
        self._img = None

    # -- helpers ------------------------------------------------------------------------------
    def _new_page(self, title: str, subtitle: str, step: int):
        if self.page is not None:
            self.page.destroy()
        self.video = None
        p = self.page = tk.Frame(self.win, bg=BG)
        p.pack(fill="both", expand=True, padx=40, pady=28)
        steps = tk.Frame(p, bg=BG)
        steps.pack(side="bottom", pady=(10, 0))
        for k, lab in enumerate(["Your words", "Practice", "Train"]):
            color = ACCENT if k == step else GREEN if k < step else "#6b6670"
            tk.Label(steps, text=("✓ " if k < step else "") + lab, fg=color, bg=BG,
                     font=(FONT, 9, "bold")).pack(side="left", padx=18)
        self.page_title = tk.Label(p, text=title, fg=FG, bg=BG, font=(FONT, 20, "bold"))
        self.page_title.pack(pady=(24, 8))
        self.page_sub = tk.Label(p, text=subtitle, fg=DIM, bg=BG, font=(FONT, 11), wraplength=WW - 100,
                                 justify="center")
        self.page_sub.pack()
        return p

    def _button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, relief="flat", cursor="hand2",
                         bg=ACCENT if primary else "#34303a", fg="white", activebackground="#d62961",
                         activeforeground="white", font=(FONT, 11, "bold" if primary else "normal"),
                         padx=22, pady=8, bd=0)

    def show(self, start_at: str = "welcome"):
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()
        if start_at == "practice":
            self.goPractice()
        else:
            self.welcome()

    def close(self):
        self.app.onboarding = None
        self.app.camera.track_always = False
        self.win.withdraw()

    # -- 1. welcome ---------------------------------------------------------------------------
    def welcome(self):
        key = self.app.key_name
        p = self._new_page("Lipflow reads your lips",
                           f"Hold {key}, mouth what you want to say without making a sound, let go, and the "
                           "text appears wherever you're typing. Setup takes about 10 minutes: it learns your "
                           "words and your face.", -1)
        tk.Label(p, text="Everything stays on this PC: the camera video, your practice clips and the models "
                         "trained on them. Nothing is uploaded.\n\nIf the camera doesn't come on, turn on "
                         "Settings → Privacy & security → Camera → \"Let desktop apps access your camera\".",
                 fg=DIM, bg=BG, font=(FONT, 10), wraplength=WW - 120, justify="center").pack(pady=26)
        self._button(p, "Get started", self.goWords, primary=True).pack(pady=12)

    # -- 2. your words (Wispr import) -----------------------------------------------------------
    def goWords(self):
        from ..personal import PHRASES, WISPR_DIR
        p = self._new_page("Teach it your words",
                           "Most of what you'll mouth is stuff you already say. If you use Wispr Flow, Lipflow "
                           "can learn your phrasing and names from its history, read locally, never uploaded.", 0)
        self.words_status = tk.Label(p, text="", fg=FG, bg=BG, font=(FONT, 11), wraplength=WW - 120)
        self.words_status.pack(pady=24)
        row = tk.Frame(p, bg=BG)
        row.pack(pady=8)
        if os.path.exists(PHRASES):
            n = sum(1 for _ in open(PHRASES, encoding="utf-8"))
            self.words_status.configure(text=f"Already imported: {n:,} of your phrases.")
        elif os.path.isdir(WISPR_DIR):
            self._button(row, "Import from Wispr Flow", self.importWispr).pack(side="left", padx=6)
        else:
            self.words_status.configure(text="Wispr Flow isn't installed here, so practice uses standard "
                                             "sentences. That works fine.")
        self._button(row, "Continue", self.goPractice, primary=True).pack(side="left", padx=6)

    def importWispr(self):
        self.words_status.configure(text="Reading your Wispr Flow history…")

        def work():
            from ..personal import import_wispr
            try:
                s = import_wispr()
                msg = f"Imported {s['phrases']:,} phrases and {len(s['new_names'])} names."
            except Exception as e:
                msg = f"Couldn't import: {e}"
            self.app.ui(self.words_status.configure, text=msg)
        threading.Thread(target=work, daemon=True).start()

    # -- 3. practice ----------------------------------------------------------------------------
    def goPractice(self):
        done = {c["text"] for c in saved_clips(self.app.opts.language)}
        self.sentences = [x for x in practice_sentences(N_SENTENCES * 2, self.app.opts.language) if x not in done][:N_SENTENCES]
        self.i = 0
        self.round_clips = []
        prior = len(done)
        p = self._new_page("Mouth each sentence",
                           f"Hold {self.app.key_name}, silently mouth the sentence at your normal pace, then let "
                           "go. Face the camera with your mouth in good light."
                           + (f" You have {prior} clips from before; these add to them." if prior else ""), 1)
        self.count = tk.Label(p, text="", fg=ACCENT, bg=BG, font=(FONT, 9, "bold"))
        self.count.pack(pady=(18, 2))
        self.prompt = tk.Label(p, text="", fg=FG, bg=BG, font=(FONT, 17), wraplength=WW - 100)
        self.prompt.pack(pady=6)
        holder = tk.Frame(p, bg="#000000", width=208, height=130)  # pixels (a bare Label counts characters)
        holder.pack_propagate(False)
        holder.pack(pady=8)
        self.video = tk.Label(holder, bg="#000000")
        self.video.pack(fill="both", expand=True)
        self.feedback = tk.Label(p, text="", fg=DIM, bg=BG, font=(FONT, 10))
        self.feedback.pack()
        row = tk.Frame(p, bg=BG)
        row.pack(pady=10)
        self._button(row, "Redo last", self.redo).pack(side="left", padx=6)
        self._button(row, "Skip sentence", self.skip).pack(side="left", padx=6)
        self.app.camera.track_always = True
        self.app.camera.ensure_open()
        self.app.onboarding = self
        self._show_sentence()

    def _show_sentence(self):
        self.count.configure(text=f"SENTENCE {self.i + 1} OF {N_SENTENCES}")
        s = self.sentences[self.i % len(self.sentences)]
        self.prompt.configure(text=s)
        self.app.onboarding_text = s

    def set_frame(self, video_bgr):
        if self.app.onboarding is self and video_bgr is not None and self.video is not None:
            self._img = photo(video_bgr)
            self.video.configure(image=self._img)

    def clip_done(self, rec_ok: bool, message: str, rois=None, text=None, raw=None):
        """Called by the app (main thread) after each practice recording."""
        if not rec_ok:
            self.feedback.configure(fg=AMBER, text=message)
            return
        self.round_clips.append(save_clip(rois, text, raw, self.app.opts.language))
        self.feedback.configure(fg=DIM, text=f"Saved. The model read: \"{(raw or '').lower()}\"")
        self.i += 1
        if self.i >= N_SENTENCES:
            self.app.onboarding = None
            return self.goTrain()
        self._show_sentence()

    def redo(self):
        if self.round_clips and self.i > 0:
            os.remove(self.round_clips.pop())
            self.i -= 1
            self.feedback.configure(fg=DIM, text="Removed the last clip. Try it again.")
            self._show_sentence()

    def skip(self):
        self.sentences.append(self.sentences.pop(self.i % len(self.sentences)))
        self._show_sentence()

    # -- 4. train -------------------------------------------------------------------------------
    def goTrain(self):
        self.app.onboarding = None
        self.app.camera.track_always = False
        gpu = self.app.reader is not None and self.app.reader.enc_device.type == "cuda"
        p = self._new_page("Learning your face",
                           ("Training runs on your graphics card" if gpu else
                            "Training runs on the processor, so it can take a while")
                           + ". You can keep working; Lipflow is paused until it's done.", 2)
        self.progress = ttk.Progressbar(p, length=WW - 160, maximum=100)
        self.progress.pack(pady=(40, 12))
        self.train_status = tk.Label(p, text="Starting…", fg=DIM, bg=BG, font=(FONT, 11), wraplength=WW - 120)
        self.train_status.pack()
        self.result = tk.Label(p, text="", fg=FG, bg=BG, font=(FONT, 13, "bold"), wraplength=WW - 120)
        self.result.pack(pady=18)
        self.done_btn = self._button(p, "Start using Lipflow", self.finish, primary=True)
        self.app.jobs.put(("train", self))

    def report(self, pct: float, text: str):  # model thread
        self.app.ui(self.progress.configure, value=pct)
        self.app.ui(self.train_status.configure, text=text)

    def finished(self, before: float, after: "float | None", kept: bool, note: str):  # model thread
        def ui():
            self.progress.configure(value=100)
            self.page_title.configure(text="You're all set" if (after is None or kept) else "Setup finished")
            self.page_sub.configure(text=f"Hold {self.app.key_name} anywhere and mouth your words. You can "
                                         "retrain any time from the tray menu: more practice makes it better.")
            if after is None:
                self.result.configure(text=note)
            else:
                self.result.configure(
                    fg=GREEN if kept else AMBER,
                    text=f"Words read correctly on sentences it didn't train on: {1 - before:.0%} → {1 - after:.0%}"
                         + ("" if kept else "\nNo improvement, so the standard model stays."))
            self.train_status.configure(text=note if after is not None else "")
            self.done_btn.pack(pady=8)
        self.app.ui(ui)

    def finish(self):
        from ..dictation import load_settings, save_settings
        s = load_settings()
        s["onboarded"] = True
        save_settings(s)
        self.app.settings["onboarded"] = True
        self.win.withdraw()
        self.app.hud.show("done", "Lipflow is ready", f"Hold {self.app.key_name} anywhere and mouth your words", 3.0)
