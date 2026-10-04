import os

from lipflow.cleanup import basic_cleanup, numbers_to_digits, _user_prompt


def test_years_and_numbers():
    assert numbers_to_digits("married in nineteen fifty two") == "married in 1952"
    assert numbers_to_digits("it is twenty twenty six") == "it is 2026"
    assert numbers_to_digits("appeared in eleven films") == "appeared in 11 films"
    assert numbers_to_digits("one dog and two cats") == "one dog and two cats"
    assert numbers_to_digits("forty two") == "42"


def test_basic_cleanup():
    assert basic_cleanup("I THINK I'LL GO") == "I think I'll go."
    assert basic_cleanup("WHAT TIME IS IT") == "What time is it?"
    assert basic_cleanup("") == ""


def test_prompt_lists_candidates_and_context():
    p = _user_prompt(["A B", "A C"], "earlier text")
    assert "1. A B" in p and "2. A C" in p and "earlier text" in p


def test_custom_words_pick_the_guess_and_fix_case():
    from lipflow.cleanup import Cleaner
    from lipflow.personal import Personal
    c = Cleaner("basic")
    c.personal = Personal("/nonexistent")  # don't depend on the user's imported history
    guesses = ["HELLO MCCALL I AM SENDING YOU A MESSAGE", "HELLO MIGUEL I AM SENDING YOU A MESSAGE"]
    assert c(guesses, words=["Miguel"]) == "Hello Miguel I am sending you a message."
    assert c(guesses, words=[]) == "Hello mccall I am sending you a message."


def test_small_model_may_not_invent_words():
    from lipflow.cleanup import within_guesses, fix_case
    guesses = ["HELLO CAN YOU EAT WHAT I'M SAYING", "HELLO CAN YOU GUESS WHAT I'M SAYING"]
    assert within_guesses("Hello, can you eat what I'm saying?", guesses, strict=True)
    assert not within_guesses("Hello, can you eat them?", guesses, strict=True)
    assert within_guesses("Hello, can you guess what I'm saying?", guesses, strict=False)
    assert not within_guesses("Hello, can you eat them?", guesses, strict=False)
    assert fix_case("hello miguel i'm here") == "Hello miguel I'm here"


def test_vocab_edit_guard():
    from lipflow.cleanup import within_guesses
    guesses = ["HELLO MIGUEL I AM SENDING YOU A MESSAGE WITH MY NEW", "HELLO MIGUEL I AM SENDING YOU A MESSAGE WITH MY NEWS"]
    known = {"tool", "deck"}.__contains__
    assert within_guesses("Hello Miguel, I am sending you a message with my new tool.", guesses, False, known, 1)
    assert not within_guesses("Hello Miguel, I am sending you a deck with my new tool.", guesses, False, known, 1)
    assert not within_guesses("Hello Miguel, I am sending you a message with my new toy.", guesses, False, known, 1)


def test_practice_sentences_fall_back_to_harvard(tmp_path, monkeypatch):
    import lipflow.personal as P
    from lipflow import practice as O
    monkeypatch.setattr(P, "PHRASES", str(tmp_path / "none.txt"))
    s = O.practice_sentences(24)
    assert len(s) == 24 and len(set(s)) == 24 and all(x in O.HARVARD for x in s)


def test_training_targets_drop_punctuation():
    from lipflow.train_vsr import _targets

    class R:
        token_list = ["<blank>"] + [l.split()[0] for l in open("lipflow/unigram5000_units.txt", encoding="utf-8").read().splitlines()] + ["<eos>"]
    r = R()
    unk = r.token_list.index("<unk>")
    assert unk not in _targets(r, "Hello, Miguel. It's done!")


def test_practice_mixes_own_and_harvard(tmp_path, monkeypatch):
    import lipflow.personal as P
    from lipflow import practice as O
    f = tmp_path / "phrases.txt"
    f.write_text("\n".join(f"This is my own sentence number {w} for testing" for w in
                           "one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split()))
    monkeypatch.setattr(P, "PHRASES", str(f))
    s = O.practice_sentences(24)
    assert sum(x in O.HARVARD for x in s) == 12 and len(set(s)) == 24


def test_names_snap_by_lip_shape_but_everyday_words_stay():
    from lipflow.visemes import snap_names
    common = {"i", "am", "a", "my", "hello", "sending", "you", "message", "with", "new", "school", "tool",
              "made", "mistake", "in", "the", "meeting"}.__contains__
    snap = lambda t: snap_names(t, ["Miguel"], lambda w: common(w.lower()))
    assert snap("HELLO MCCALL I AM SENDING YOU A MESSAGE") == "HELLO MIGUEL I AM SENDING YOU A MESSAGE"
    assert snap("HELLO MC HALE I AM SENDING") == "HELLO MIGUEL I AM SENDING"
    assert snap("HELLO MIKAEL") == "HELLO MIGUEL"
    assert snap("I MADE A MISTAKE IN THE MEETING") == "I MADE A MISTAKE IN THE MEETING"


def test_context_names_from_titles():
    from lipflow.context import extract_names
    assert extract_names("Miguel (DM) - Vizcom - Slack") == ["Miguel", "Vizcom", "Slack"]
    assert "Priya" in extract_names("Re: design review", "Thanks Priya, I think the plan works.")


def test_snapping_leaves_every_harvard_sentence_alone():
    from lipflow.practice import HARVARD
    from lipflow.visemes import snap_names
    for h in HARVARD:
        assert snap_names(h.upper(), ["Priya", "Miguel", "Vizcom", "Balance", "Flow"], lambda w: False) == h.upper()



def test_codex_runs_read_only_and_reads_the_last_message(monkeypatch):
    """codex exec is an agent: it must run sandboxed, in an empty folder, and only its final reply is used."""
    import subprocess
    from lipflow import cleanup
    from lipflow.cleanup import Cleaner
    monkeypatch.setattr(cleanup.shutil, "which", lambda name: "/usr/local/bin/codex")
    calls = []

    def run(args, **kw):
        calls.append((args, kw))
        out = args[args.index("-o") + 1]
        open(out, "w").write("Hi Priya, can we meet at 4:30?\n")
        return subprocess.CompletedProcess(args, 0, "progress noise", "")

    monkeypatch.setattr(cleanup.subprocess, "run", run)
    c = Cleaner("codex")
    assert c.backend == "codex"
    assert c(["HI PRE A CAN WE MEET AT FOUR THIRTY"], words=[]) == "Hi Priya, can we meet at 4:30?"
    args, kw = calls[0]
    assert args[:2] == ["codex", "exec"] and args[-1] == "-"
    assert args[args.index("--sandbox") + 1] == "read-only" and "--ephemeral" in args
    assert kw["cwd"] != os.getcwd() and kw["timeout"] <= 20
    assert "HI PRE A CAN WE MEET AT FOUR THIRTY" in kw["input"]


def test_codex_failure_falls_back_to_basic(monkeypatch):
    import subprocess
    from lipflow import cleanup
    from lipflow.cleanup import Cleaner
    monkeypatch.setattr(cleanup.shutil, "which", lambda name: "/usr/local/bin/codex")
    monkeypatch.setattr(cleanup.subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "Not logged in"))
    assert Cleaner("codex")(["WHAT TIME IS IT"], words=[]) == "What time is it?"

    def timeout(args, **kw):
        raise subprocess.TimeoutExpired(args, kw["timeout"])
    monkeypatch.setattr(cleanup.subprocess, "run", timeout)
    assert Cleaner("codex")(["WHAT TIME IS IT"], words=[]) == "What time is it?"


def test_backend_is_your_choice(monkeypatch):
    """Automatic never picks Codex; a choice that isn't set up falls back to automatic."""
    from lipflow import cleanup
    from lipflow.cleanup import Cleaner
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(Cleaner, "_ollama_up", staticmethod(lambda: False))
    monkeypatch.setattr(cleanup.shutil, "which", lambda name: "/usr/local/bin/codex")
    assert Cleaner("auto").backend != "codex"
    assert Cleaner("codex").backend == "codex"
    assert cleanup.unavailable("claude") == "set ANTHROPIC_API_KEY"
    assert Cleaner("claude").backend == Cleaner("auto").backend
    monkeypatch.setattr(cleanup.shutil, "which", lambda name: None)
    assert cleanup.unavailable("codex") and Cleaner("codex").backend == Cleaner("auto").backend


def test_on_device_model_needs_apple_silicon(monkeypatch):
    from lipflow import cleanup
    monkeypatch.setattr(cleanup.sys, "platform", "darwin")
    monkeypatch.setattr(cleanup.platform, "machine", lambda: "x86_64")
    assert cleanup.unavailable("local") == "Apple Silicon only"
    monkeypatch.setattr(cleanup.platform, "machine", lambda: "arm64")
    assert cleanup.unavailable("local") is None


def test_intel_mac_encoder_stays_on_cpu(monkeypatch):
    from lipflow import vsr
    monkeypatch.setattr(vsr.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(vsr.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(vsr.platform, "machine", lambda: "x86_64")
    assert vsr.pick_encoder_device().type == "cpu"
    monkeypatch.setattr(vsr.platform, "machine", lambda: "arm64")
    assert vsr.pick_encoder_device().type == "mps"
