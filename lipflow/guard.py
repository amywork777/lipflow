"""Backend-independent checks for changes that need human review, including Chinese."""
from collections import Counter
import re

from .text import tokens, contains

NEG = re.compile(r"\b(?:no|not|never|neither|without|cannot|can't|don't|doesn't|didn't|won't|isn't|aren't|shouldn't|wouldn't|couldn't|mustn't)\b|不要|不能|不用|没有|不是|不会|不|没|勿|未", re.I)
NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*|[零〇一二两三四五六七八九十百千万亿]+")
# Units/relative dates are intentionally conservative: formatting that changes them needs review.
UNITS = re.compile(r"[$€£¥￥%]|\b(?:dollars?|euros?|pounds?|percent|today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b|今天|明天|昨天|后天|前天|元|块|美元|人民币|年|月|日|号|点|时|分|秒|百分之", re.I)


def edits(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j]+1, d[j-1]+1, prev+(x != y))
    return d[-1]


def check(original: str, proposed: str, candidates=(), names=(), mode='faithful') -> tuple[str, ...]:
    from .cleanup import numbers_to_digits
    a, b = tokens(original), tokens(proposed)
    if not b:
        return ('Empty cleanup output',)
    warnings = []
    # English number formatting is normalized identically on both sides.
    left, right = numbers_to_digits(original.casefold()), numbers_to_digits(proposed.casefold())
    if Counter(NUMBER.findall(left)) != Counter(NUMBER.findall(right)):
        warnings.append('Numbers changed / 数字发生变化')
    if Counter(UNITS.findall(left)) != Counter(UNITS.findall(right)):
        warnings.append('Date, time or amount changed / 日期、时间或金额发生变化')
    if Counter(NEG.findall(left)) != Counter(NEG.findall(right)):
        warnings.append('Negation changed / 否定表达发生变化')
    if any(contains(original, n) != contains(proposed, n) for n in names if n):
        warnings.append('Name or custom term changed / 姓名或术语发生变化')
    if a == b and not warnings:
        return ()
    a, b = tokens(left), tokens(right)
    distance = edits(a, b)
    if mode == 'polish':
        warnings.append('Polished wording requires review / 润色结果需确认')
    elif distance > max(1, int(len(a) * 0.2)):
        warnings.append('Large wording change / 改动过大')
    pool = {t for c in (original, *candidates) for t in tokens(numbers_to_digits(c.casefold()))}
    if mode == 'faithful' and any(t not in pool for t in b):
        warnings.append('Words outside recognition candidates / 添加了识别候选以外的内容')
    return tuple(warnings)
