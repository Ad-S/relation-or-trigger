"""Trigger lexicon for the interpretability analysis.

SELECTION = the phrases used to mine candidates (relational-fallacy-detection/finetune/triggers.py, unchanged).
EXTENDED adds close variants of the same denial / attribution moves that the selection list misses
("you're saying" without "that", "I'm just saying", "what I mean"...). The extended list defines the
`trigger` probe target, the trigger token span, and is what the trigger-swap (ts) edits toggle.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "relational-fallacy-detection" / "finetune"))
from triggers import PHRASES as SELECTION  # noqa: E402

EXTRA = {
    "motte_bailey": ["i'm just saying", "i am just saying", "i'm saying", "i am saying", "what i mean", "i just mean",
                     "i only mean", "i never claimed", "i didn't claim", "i'm not claiming", "my point was",
                     "what i'm trying to get at", "i didn't mean", "i never meant", "i meant"],
    "strawman": ["you're saying", "you are saying", "your argument is", "you seem to be saying", "what you seem to be saying",
                 "are you saying", "it seems like you're saying", "you're implying", "are you implying", "you're actually saying"],
}
EXTENDED = {s: sorted(set(SELECTION[s]) | set(EXTRA[s]), key=len, reverse=True) for s in SELECTION}
PATTERN = {s: re.compile("|".join(re.escape(p) for p in EXTENDED[s]).replace("'", "['’]"), re.I) for s in EXTENDED}


def find_triggers(text, shape):
    """Character spans (start, end) of every extended-lexicon trigger in `text`."""
    return [m.span() for m in PATTERN[shape].finditer(text)]


def has_trigger(text, shape):
    return bool(PATTERN[shape].search(text))
