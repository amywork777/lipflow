"""Unicode-aware tokens: English words and individual Han characters (CER units)."""
import re
import unicodedata

HAN = r"\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f"
TOKEN = re.compile(rf"[{HAN}]|[a-z0-9]+(?:'[a-z]+)?", re.I)


def has_han(text: str) -> bool:
    return bool(re.search(rf"[{HAN}]", text))


def tokens(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize('NFKC', text).casefold().replace('’', "'"))


def contains(text: str, term: str) -> bool:
    if has_han(term):
        return term.casefold() in text.casefold()
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text, re.I))
