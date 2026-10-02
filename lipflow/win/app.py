"""Lipflow for Windows: a tray icon. Hold a key, mouth the words, let go — the text appears at your cursor.

Threads: tk owns the main thread (overlay, setup window); pynput's hook thread reports the key;
pystray runs the tray menu on its own thread; one model thread reads lips. Everything that touches
tk goes through ui(), which queues it for the main thread.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

from ..camera import Camera, Recording, mouth_view
from ..cleanup import Cleaner
from ..dictation import (
    HISTORY, JOIN_WINDOW, MAX_SECONDS, PREVIEW_EVERY, TAIL_SECONDS, clip_problem, keep_clip, load_settings,
    log_history, rois_for, save_settings, train_on_face,
)
from ..paths import HOME
from ..vsr import LipReader
from .hotkey import DEFAULT_KEY, KEYS, PushToTalk
from .hud import HUD, tray_image
from .paste import copy_text, paste_text

LOG = os.path.join(HOME, "Lipflow.log")
CAMERAS = 4  # Windows can't name cameras through OpenCV: offer the first few by number
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


@dataclass
class Options:
    key: str = DEFAULT_KEY
    beam: int = 4
    backend: str = "auto"
    camera: "int | str" = "auto"
    paste: bool = True
    live_preview: bool = True
    onboard: bool = False
    language: str | None = None
    cleanup_mode: str | None = None
    confidence_policy: str | None = None
    min_margin: float = 0.5
    input_mode: str | None = None


def key_label(key: str) -> str:
    return key.replace("_", " ").title().replace("Control", "Ctrl")


class Lipflow:
    def __init__(self, opts: Options):
        self.opts = opts
        self.settings = load_settings()
        if opts.key == DEFAULT_KEY and self.settings.get("key") in KEYS:
            opts.key = self.settings["key"]
        self.root = tk.Tk()
        self.root.withdraw()
        self._q: "queue.Queue" = queue.Queue()
        self.reader: LipReader | None = None
        settings = load_settings()
        opts.language = opts.language or settings.get("language", "en")
        opts.cleanup_mode = opts.cleanup_mode or settings.get("cleanup_mode", "faithful")
        opts.confidence_policy = opts.confidence_policy or settings.get("confidence_policy", "review")
        opts.input_mode = opts.input_mode or settings.get("input_mode", "whisper" if settings.get("whisper") else "silent")
        self.cleaner = Cleaner(opts.backend, opts.cleanup_mode, opts.language)
        self.settings["whisper"] = opts.input_mode == "whisper"
        self.last_raw = ""
        self.review_pending = False
        self.jobs: "queue.Queue" = queue.Queue()
        self.session = 0          # bumps on every start/cancel so stale previews are dropped
        self.preview_busy = False
        self.last_output = ""
        self.last_paste_at = 0.0
        self.context: list[str] = []
        self.hands_free = False
        self.pending_stop = None
        from ..mic import Mic
        self.mic = Mic()
        self.av_reader = None
        self._ui_busy = False
        self.onboarding = None
        self.onboarding_text = ""
        self.setup = None
        self.loading = True
        self.state_text = "Loading model…"
        self.ctx = None
        cam = opts.camera if opts.camera != "auto" else self.settings.get("camera", "auto")
        self.camera = Camera(cam, on_frame=self.on_frame)

    @property
    def key_name(self) -> str:
        return key_label(self.opts.key)

    # -- threading -------------------------------------------------------------------------
    def ui(self, fn, *args, **kw):
        """Run fn on the tk thread (safe from any thread)."""
        self._q.put((fn, args, kw))

    def _pump(self):
        try:
            while True:
                fn, args, kw = self._q.get_nowait()
                try:
                    fn(*args, **kw)
                except Exception:
                    import traceback
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(15, self._pump)

    # -- setup -----------------------------------------------------------------------------
    def start(self):
        self.hud = HUD(self.root)
        from .review import Review
        self.review = Review(self.root)
        self._install_key()
        self._build_tray()
        self.hud.show("reading", "Lipflow", "Loading the lip-reading model…")
        threading.Thread(target=self._worker, name="lipflow-model", daemon=True).start()
        self.jobs.put(("load",))
        self.root.after(15, self._pump)

    def _install_key(self):
        # The hook thread only queues: all state changes happen on the tk thread.
        self.ptt = PushToTalk(self.opts.key,
                              lambda hands_free: self.ui(self.on_start, hands_free),
                              lambda: self.ui(self.on_stop),
                              lambda silent=False: self.ui(self.on_cancel, silent))
        self.ptt.install()

    def _build_tray(self):
        import pystray
        from pystray import Menu, MenuItem as Item

        def toggle(name, default=True, then=None):
            def act(icon, item):
                self.settings[name] = not self.settings.get(name, default)
                save_settings(self.settings)
                if then:
                    then()
            return Item(name_labels[name], act, checked=lambda item: self.settings.get(name, default),
                        enabled=lambda item: name != "whisper" or self.opts.language == "en")

        name_labels = {"whisper": "Whisper mode (lips + a soft whisper)",
                       "use_context": "Use the window title for names",
                       "save_clips": "Keep my last 100 clips (local)"}

        def pick_camera(value):
            return Item("Automatic" if value == "auto" else f"Camera {value + 1}",
                        lambda icon, item: self.ui(self._pick_camera, value),
                        checked=lambda item: self.settings.get("camera", "auto") == value, radio=True)

        def pick_key(name):
            return Item(key_label(name), lambda icon, item: self.ui(self._pick_key, name),
                        checked=lambda item: self.opts.key == name, radio=True)

        def preference(title, key, items):
            return Item(title + " (restart)", Menu(*[
                Item(label, lambda icon, item, k=key, v=value: self.ui(self._preference, k, v),
                     checked=lambda item, k=key, v=value: self.settings.get(k, getattr(self.opts, k)) == v,
                     radio=True) for label, value in items]))

        menu = Menu(
            Item(lambda item: self.state_text, None, enabled=False),
            Item(lambda item: f"Hold {self.key_name} to dictate, double-tap for hands-free", None, enabled=False),
            Item(lambda item: f"Cleanup: {self.cleaner.describe()}", None, enabled=False),
            Menu.SEPARATOR,
            Item("Copy raw recognition", lambda icon, item: self.ui(copy_text, self.last_raw)),
            Item("Copy last dictation", lambda icon, item: self.ui(self._copy_last)),
            Item("Practice && train more…", lambda icon, item: self.ui(self.show_setup, "practice")),
            Item("Run setup again…", lambda icon, item: self.ui(self.show_setup)),
            Menu.SEPARATOR,
            preference("Language / 识别语言", "language", [("English", "en"), ("中文 · CMLR research", "zh")]),
            preference("Cleanup mode", "cleanup_mode", [("Faithful / 忠实", "faithful"), ("Polish / 润色", "polish")]),
            preference("Candidate policy", "confidence_policy", [("Review", "review"), ("Heuristic auto", "auto")]),
            preference("Input mode", "input_mode", [("Silent", "silent"), ("Whisper", "whisper")]),
            Item("Camera", Menu(*[pick_camera(v) for v in ["auto", *range(CAMERAS)]])),
            Item("Push-to-talk key", Menu(*[pick_key(k) for k in KEYS])),
            toggle("whisper", False, then=lambda: self.ui(self._whisper_changed)),
            toggle("use_context"),
            toggle("save_clips"),
            Item("Start with Windows", lambda icon, item: self._toggle_autostart(),
                 checked=lambda item: self._autostart_enabled()),
            Menu.SEPARATOR,
            Item("Open history", lambda icon, item: self._notepad(HISTORY)),
            Item("Edit custom words…", lambda icon, item: self._edit_words()),
            Item("Open log", lambda icon, item: self._notepad(LOG)),
            Menu.SEPARATOR,
            Item("Quit Lipflow", lambda icon, item: self.ui(self.quit)),
        )
        self.icon = pystray.Icon("Lipflow", tray_image(False), "Lipflow", menu)
        threading.Thread(target=self.icon.run, name="lipflow-tray", daemon=True).start()

    def _set_icon(self, listening: bool):
        try:
            self.icon.icon = tray_image(listening)
        except Exception:
            pass

    def _set_state(self, text: str):
        self.state_text = text
        try:
            self.icon.update_menu()
        except Exception:
            pass

    # -- menu actions (tk thread unless noted) ---------------------------------------------------
    def _preference(self, key, value):
        self.settings[key] = value
        save_settings(self.settings)
        self.icon.update_menu()
        self.hud.show("done", "Saved / 已保存", "Restart Lipflow to apply / 重启后生效", 4.0)

    def _copy_last(self):
        if self.last_output:
            copy_text(self.last_output)

    def _pick_camera(self, value):
        self.settings["camera"] = value
        save_settings(self.settings)
        self.camera.set_source(value)
        self.icon.update_menu()  # pystray rebuilt it before this queued change ran
        print(f"[lipflow] camera: {value}")

    def _pick_key(self, name):
        self.ptt.stop()
        self.opts.key = name
        self.settings["key"] = name
        save_settings(self.settings)
        self._install_key()
        self.icon.update_menu()
        self.hud.show("done", "Push-to-talk key", f"Hold {self.key_name} to dictate", 2.0)

    def _whisper_changed(self):
        self.opts.input_mode = "whisper" if self.settings.get("whisper") else "silent"
        self.settings["input_mode"] = self.opts.input_mode
        save_settings(self.settings)
        if self.settings.get("whisper") and self.av_reader is None and not self.loading:
            self.jobs.put(("whisper",))

    @staticmethod
    def _notepad(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a", encoding="utf-8").close()
        subprocess.Popen(["notepad.exe", path])

    def _edit_words(self):
        from .. import vocab
        vocab.load()
        self._notepad(vocab.PATH)

    @staticmethod
    def _autostart_command() -> str:
        exe = sys.executable
        if exe.lower().endswith("python.exe"):  # no console window at login
            exe = exe[:-len("python.exe")] + "pythonw.exe"
        return f'"{exe}" -X utf8 -m lipflow'

    def _autostart_enabled(self) -> bool:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
                winreg.QueryValueEx(k, "Lipflow")
                return True
        except OSError:
            return False

    def _toggle_autostart(self):  # tray thread; registry only
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            if self._autostart_enabled():
                winreg.DeleteValue(k, "Lipflow")
            else:
                winreg.SetValueEx(k, "Lipflow", 0, winreg.REG_SZ, self._autostart_command())

    def show_setup(self, start_at: str = "welcome"):
        if self.opts.language == "zh" and self.opts.input_mode == "whisper":
            self.hud.show("error", "Lip training / 唇读训练", "Switch to silent mode and restart / 请切换无声模式并重启", 4.0)
            return
        from .setup import Setup
        if self.loading and start_at == "practice":
            self.hud.show("error", "Still loading", "Try again in a moment", 1.5)
            return
        if self.setup is None:
            self.setup = Setup(self)
        self.setup.show(start_at)

    def quit(self):
        self.camera.close()
        self.ptt.stop()
        try:
            self.icon.stop()
        except Exception:
            pass
        self.root.destroy()

    # -- push-to-talk (tk thread) -------------------------------------------------------------
    def on_start(self, hands_free: bool):
        if self.review_pending:
            return
        if self.loading:
            self.hud.show("error", "Still loading", "The model is almost ready…", hide_after=1.5)
            return
        if self.pending_stop is not None:  # pressed again during the tail: finish the last one now
            self._finish_stop(self.pending_stop)
        if hands_free and self.camera.recording is not None:
            self.hands_free = True  # the second tap of a double-tap: keep the recording going
            self.hud.show("listening", "Hands-free · tap to finish", self.hud.body_text)
            return
        self.session += 1
        self.hands_free = hands_free
        from ..context import Context, capture
        # the app you're typing into is in front right now
        self.ctx = capture()
        if not self.settings.get("use_context", True):
            self.ctx.names = []
        rec = self.camera.start_recording()
        rec.ctx = self.ctx
        if self.whisper_on:
            self.mic.start()
        self._set_icon(True)
        title = "Hands-free · tap to finish" if hands_free else "Listening"
        self.hud.show("listening", title, "" if self.camera.ready.is_set() else "Starting camera…")
        threading.Thread(target=self._preview_loop, args=(self.session, rec), daemon=True).start()

    def on_stop(self):
        self.hands_free = False
        if self.camera.recording is None:
            return
        self.session += 1
        self.pending_stop = self.session
        self._set_icon(False)
        self.hud.show("reading", "Reading your lips", self.hud.body_text)
        token = self.session
        self.root.after(int(TAIL_SECONDS * 1000), lambda: self._finish_stop(token))

    def _finish_stop(self, token):
        if self.pending_stop != token:
            return
        self.pending_stop = None
        rec = self.camera.stop_recording()
        audio = self.mic.stop() if self.whisper_on else []
        if rec is not None:
            rec.audio = audio
            rec.session = token
            self.jobs.put(("final", rec))

    def on_cancel(self, silent: bool = False):
        self.review.dismiss()
        self.review_pending = False
        self.pending_stop = None
        self.mic.stop()
        self._set_icon(False)
        self.session += 1
        self.hands_free = False
        self.camera.stop_recording()
        if silent:
            self.hud.hide()
        else:
            self.hud.show("error", "Cancelled", "", hide_after=0.8)

    # -- camera thread ------------------------------------------------------------------------
    def on_frame(self, frame, obs, recording):
        """At most one video frame waits for the tk thread at a time, so frames never pile up in
        front of the key handling."""
        rec = self.camera.recording
        if recording and rec is not None and rec.duration > MAX_SECONDS:
            self.ui(self.on_stop)
            return
        if self._ui_busy or (not recording and self.onboarding is None):
            return
        setup = mouth_view(frame, obs, 208, 130) if self.onboarding is not None else None
        pill = mouth_view(frame, obs, 112, 70) if recording else None
        self._ui_busy = True
        self.ui(self._show_frame, setup, pill)

    def _show_frame(self, setup, pill):
        try:
            if setup is not None and self.onboarding is not None:
                self.onboarding.set_frame(setup)
            if pill is not None:
                self.hud.set_frame(pill)
        finally:
            self._ui_busy = False

    # -- model thread -------------------------------------------------------------------------
    def _preview_loop(self, session: int, rec: Recording):
        if not self.opts.live_preview or (self.opts.language == "zh" and self.opts.input_mode == "whisper"):
            return
        waited = 0.0
        while self.session == session:
            time.sleep(PREVIEW_EVERY)
            waited += PREVIEW_EVERY
            if not rec.ts and (self.camera.error or waited > 6):
                msg = self.camera.error or "The camera isn't sending frames"
                print(f"[lipflow] camera problem: {msg}")
                self.ui(self.hud.show, "error", "Camera problem", msg, 6.0)
                return
            if self.session != session or self.preview_busy or len(rec.ts) < 15:
                continue
            self.preview_busy = True
            self.jobs.put(("preview", session, rec))

    def _worker(self):
        while True:
            job = self.jobs.get()
            try:
                if job[0] == "load":
                    self._load()
                elif job[0] == "preview":
                    self._preview(*job[1:])
                elif job[0] == "final":
                    self._final(job[1])
                elif job[0] == "train":
                    self._train(job[1])
                elif job[0] == "whisper":
                    self._load_whisper()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.ui(self.hud.show, "error", "Something went wrong", str(e)[:80], 3.0)
            finally:
                if job[0] == "preview":
                    self.preview_busy = False

    def _load(self):
        t = time.time()
        if self.opts.language == "zh" and self.opts.input_mode == "whisper":
            self.reader = None  # quiet-speech ASR does not require research-only CMLR weights
        else:
            self.reader = LipReader(beam_size=self.opts.beam, language=self.opts.language)
            self.reader.warmup()
        if self.opts.language == "zh" and self.opts.input_mode == "whisper":
            from ..whisper import ChineseWhisper
            self.ui(self.hud.set_text, "Loading Chinese quiet-speech model (first run downloads ~1.6 GB)…")
            self.zh_whisper = ChineseWhisper()
            self.zh_whisper.load()
        self.loading = False
        print(f"[lipflow] model ready in {time.time() - t:.1f}s "
              f"(encoder on {self.reader.enc_device if self.reader else 'CPU quiet-speech'}, cleanup: {self.cleaner.describe()})")
        self.ui(self._set_state, "Ready")
        if self.settings.get("whisper") and self.opts.language == "en":
            self.jobs.put(("whisper",))
        if self.opts.language == "en" and (self.opts.onboard or not self.settings.get("onboarded")):
            self.ui(self.hud.hide)
            self.ui(self.show_setup)
        else:
            if self.opts.language == "zh" and self.opts.input_mode == "silent":
                self.ui(self.hud.show, "done", "中文唇读测试模式", "自由句子识别尚未通过验收，请逐条核对候选", 6.0)
            else:
                self.ui(self.hud.show, "done", "Lipflow is ready", f"Hold {self.key_name} and mouth your words", 2.5)

    @property
    def whisper_on(self) -> bool:
        return ((self.opts.input_mode == "whisper" and self.opts.language == "zh") or
                (self.opts.language == "en" and bool(self.settings.get("whisper")) and self.av_reader is not None)) and self.onboarding is None

    def _load_whisper(self):
        if self.opts.language != "en":
            self.ui(self.hud.show, "error", "English AV model", "Use --language zh --input-mode whisper for Chinese audio", 5.0)
            return
        from .. import av
        if not av.available():
            self.ui(self.hud.show, "reading", "Whisper mode", "Downloading the audio-visual model (1.8 GB)…")
            try:
                av.download(lambda pct: self.ui(self.hud.set_text, f"Downloading the audio-visual model… {pct:.0f}%"))
            except Exception as e:
                self.ui(self.hud.show, "error", "Whisper mode", f"Download failed: {e}"[:80], 5.0)
                return
        self.ui(self.hud.show, "reading", "Whisper mode", "Loading…")
        self.av_reader = av.AVReader(beam_size=self.opts.beam)
        self.av_reader.warmup_av()
        print("[lipflow] whisper mode ready (lips + audio)")
        self.ui(self.hud.show, "done", "Whisper mode on", "Whisper or speak softly while you mouth the words", 3.0)

    def _av_candidates(self, rec, rois):
        from ..mic import segment
        if not self.whisper_on or not getattr(rec, "audio", None):
            return None
        ts, _, _ = rec.snapshot()
        wave = segment(rec.audio, ts[0], rois.shape[0])
        if wave is None:
            return None
        return self.av_reader.hypotheses(self.av_reader.encode_av(rois, wave), nbest=5)

    def _preview(self, session: int, rec: Recording):
        if self.session != session:
            return
        rois = rois_for(rec)
        if rois is None:
            self.ui(self.hud.set_text, "Can't see your face…")
            return
        text = self.reader.greedy(self.reader.encode(rois))
        if self.session == session and text:
            self.ui(self.hud.set_text, text.lower())

    def _final(self, rec: Recording):
        t0 = time.time()
        ob = self.onboarding  # the setup window can be closed meanwhile on the tk thread
        problem = clip_problem(rec)
        if problem and ob is not None:
            print(f"[lipflow] practice clip rejected ({rec.duration:.1f}s, {len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            self.ui(ob.clip_done, False, f"{problem[0]}. {problem[1]}.")
            self.ui(self.hud.hide)
            return
        if problem:
            print(f"[lipflow] skipped {rec.duration:.1f}s clip ({len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            self.ui(self.hud.show, "error", problem[0], problem[1], 2.2)
            return
        rois = rois_for(rec)
        if rois is None:
            self.ui(self.hud.show, "error", "No face", "Face the camera / 请正对摄像头", 3.0)
            return
        enc = (None if self.opts.language == "zh" and self.opts.input_mode == "whisper" and
               self.onboarding is None else self.reader.encode(rois))
        t_enc = time.time() - t0
        if ob is not None:  # practice clip: keep it with its known text, don't paste
            raw = self.reader.greedy(enc)
            print(f"[lipflow] practice clip saved ({rec.duration:.1f}s): {raw!r}")
            self.ui(ob.clip_done, True, "", rois, self.onboarding_text, raw)
            self.ui(self.hud.hide)
            return
        hypotheses = self.reader.hypotheses(enc, nbest=5) if enc is not None else []
        if self.opts.language == "en":
            av_hypotheses = self._av_candidates(rec, rois)
            if av_hypotheses:
                hypotheses = av_hypotheses
        elif self.opts.input_mode == "whisper":
            from ..whisper import ChineseWhisper
            from ..mic import segment
            ts, _, _ = rec.snapshot()
            wave = segment(getattr(rec, "audio", []), ts[0], len(rois))
            if not hasattr(self, "zh_whisper"):
                self.zh_whisper = ChineseWhisper()
            hypotheses = self.zh_whisper.hypotheses(wave)
        greedy = self.reader.greedy(enc) if self.opts.input_mode != "whisper" else ""
        from ..delivery import choose_result, quality_for
        ctx = getattr(rec, "ctx", None)
        decision, result, choices = choose_result(hypotheses, greedy, quality_for(rec, rois),
            self.cleaner, ctx, " ".join(self.context[-3:]), self.opts.confidence_policy, self.opts.min_margin)
        self.ui(self._deliver, rec, rois, hypotheses, decision, result, choices, time.time()-t0)

    def _deliver(self, rec, rois, hypotheses, decision, result, choices, latency):
        if self.session != rec.session:
            return  # cancelled or superseded while the model was running
        if decision.action == "retry":
            self.hud.show("error", "Please repeat / 请重说", decision.reason, 5.0)
            return
        self.last_raw = result.raw
        if decision.action == "review" or result.needs_review:
            self.review_pending = True
            reason = "\n".join([decision.reason, *result.warnings, "Target: " + rec.ctx.describe()])
            self.hud.hide()
            self.review.show(choices, reason, lambda text: self._selected(rec, rois, hypotheses,
                decision, result, latency, text))
        else:
            self._publish(rec, rois, hypotheses, decision, result, latency, result.text)

    def _selected(self, rec, rois, hypotheses, decision, result, latency, text):
        self.review_pending = False
        if text is None or self.session != rec.session:
            return
        from ..delivery import restore_target
        if self.opts.paste:
            restore_target(rec.ctx)
        self.root.after(180, lambda: self._publish(rec, rois, hypotheses, decision, result, latency, text))

    def _publish(self, rec, rois, hypotheses, decision, result, latency, text):
        if self.session != rec.session:
            return
        from ..delivery import target_is_current, quality_for
        ctx = rec.ctx
        if self.opts.paste and not target_is_current(ctx):
            copy_text(text)
            self.hud.show("done", "Copied / 已复制", "Target changed; paste manually / 输入位置已变，请手动粘贴", 5.0)
            self.last_output = text
            return
        out = text
        if self.opts.language == "en" and self.last_paste_at and time.time()-self.last_paste_at < JOIN_WINDOW:
            out = " " + text
        self.last_output = text
        self.last_paste_at = time.time()
        self.context.append(text)
        candidates = [h.text for h in hypotheses]
        log_history(rec, candidates, text, latency, self.cleaner.describe(),
                    evidence={"hypotheses": [{"text": h.text, "score": h.score, "token_count": h.token_count}
                                              for h in hypotheses], "assessment": decision.asdict(),
                              "quality": vars(quality_for(rec, rois)),
                              "cleanup_proposed": result.proposed, "warnings": result.warnings,
                              "language": self.opts.language})
        keep_clip(rois, candidates, text, self.settings)
        (paste_text if self.opts.paste else copy_text)(out if self.opts.paste else text)
        self.hud.show("done", "Pasted" if self.opts.paste else "Copied", text, 2.4)

    def _train(self, ob):
        self.loading = True
        self.ui(self._set_state, "Training on your face…")
        r = train_on_face(self.opts.beam, ob.report, self.opts.language)
        if r["after"] is not None:
            self.reader = LipReader(beam_size=self.opts.beam, language=self.opts.language)
            self.reader.warmup()
            self.settings["training" if self.opts.language == "en" else "training_zh"] = {"before": r["before"], "after": r["after"], "kept": r["kept"],
                                         "clips": r["clips"], "at": time.time()}
            save_settings(self.settings)
        self.loading = False
        self.ui(self._set_state, "Ready")
        ob.finished(r["before"], r["after"], r["kept"], r["note"])


def _log_to_file():
    """pythonw has no console: send prints and tracebacks to %APPDATA%\\Lipflow\\Lipflow.log."""
    os.makedirs(HOME, exist_ok=True)
    try:
        if os.path.getsize(LOG) > 5_000_000:
            os.replace(LOG, LOG + ".old")
    except OSError:
        pass
    f = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f
    print(f"\n[lipflow] started {time.strftime('%Y-%m-%d %H:%M:%S')}")


def _already_running() -> bool:
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _already_running.handle = kernel32.CreateMutexW(None, False, "Local\\LipflowTray")  # held until exit
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def run(opts: Options):
    if sys.stdout is None or os.environ.get("LIPFLOW_APP"):
        _log_to_file()
    if _already_running():
        import ctypes
        ctypes.WinDLL("user32").MessageBoxW(None, "Lipflow is already running. Look for the pink mouth icon "
                                                  "in the system tray (you may need to click ^ to see it).",
                                            "Lipflow", 0x40)
        return
    lf = Lipflow(opts)
    lf.start()
    import signal
    signal.signal(signal.SIGINT, lambda *a: lf.ui(lf.quit))  # tk would swallow Ctrl-C in its callbacks
    print(f"[lipflow] hold {lf.key_name} and mouth your words · double-tap for hands-free · Esc cancels · Ctrl-C quits")
    lf.root.mainloop()
    os._exit(0)  # the hook, tray and model threads don't need a clean shutdown
