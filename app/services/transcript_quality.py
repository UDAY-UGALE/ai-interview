"""Decides whether a transcript is real speech worth answering.

This sits between STT and the question gate. It exists because every
speech-to-text model -- Whisper, Parakeet, Conformer, all of them -- returns
TEXT rather than nothing when handed audio that is mostly silence or noise.
The observed output from a real session was 49x "Thank you.", 15x ".", 11x
"Ct.", 7x "Tch." -- none of which anybody said. Those fragments then got
merged into the next real question and changed what the LLM was asked.

Three independent signals are used, because none of them alone is enough:

* how much of the segment was actually voiced (from the VAD)
* the model's own confidence (avg_logprob / provider confidence)
* the shape of the text itself

The text-shape rules are deliberately conservative about SHORT input: a bare
"Why?" is a legitimate follow-up and must survive, while "Ct-4, P." must
not. The distinguishing feature is not length, it is whether the tokens look
like words at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Phrases every Whisper-family model emits on silence. These are only
# treated as hallucinations when the audio backs that up (little voiced
# audio, or low confidence) -- an interviewer really can say "thank you",
# and the gate has its own small-talk handling for that case.
_SILENCE_HALLUCINATIONS = frozenset(
    {
        "thank you",
        "thanks",
        "thank you.",
        "thank you very much",
        "thanks for watching",
        "thanks for watching!",
        "you",
        "bye",
        "bye.",
        ".",
        "..",
        "...",
        "so",
        "so,",
        "okay",
        "ok",
        "oh",
        "hmm",
        "mm",
        "mhm",
        "uh",
        "um",
        "yeah",
        "the",
        "and",
        "subtitles by the amara.org community",
        "transcription by castingwords",
    }
)

# Short words with no vowel that are still real speech, so the "no vowel
# means it is not a word" rule does not eat them.
_VOWELLESS_REAL_WORDS = frozenset({"hmm", "mm", "shh", "psst", "tsk", "grr", "nth"})

# A word is a run of LETTERS IN ANY SCRIPT, optionally carrying apostrophes.
# `[^\W\d_]` is "word character that is not a digit or underscore", i.e. a
# letter in any alphabet. The shape deliberately mirrors the old ASCII-only
# `[a-z][a-z'’]*`: leading letter, then letters or apostrophes, so English
# tokenisation -- "don't" as ONE token, not "don" + "t" -- is unchanged.
_WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’])*", re.UNICODE)
_VOWEL_RE = re.compile(r"[aeiouy]", re.IGNORECASE)
# Does this token consist only of Latin letters/apostrophes? The vowel test
# below is an assertion about the LATIN alphabet and is meaningless anywhere
# else: Devanagari and Tamil write vowels as diacritics on a consonant, and
# CJK has no alphabetic vowels at all. Every such token therefore scored zero
# and any transcript written in those scripts was dropped as "no_real_words"
# before it could reach the question gate -- total, silent deafness for any
# non-Latin language. This predicate is what confines the vowel rule to the
# alphabet it was written for, leaving English behaviour bit-identical.
_LATIN_TOKEN_RE = re.compile(r"^[a-z'’]+$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class QualityVerdict:
    keep: bool
    reason: str

    def __bool__(self) -> bool:  # so `if verdict:` reads naturally
        return self.keep


def looks_like_stt_hallucination(text: str, *, min_repeats: int = 3) -> bool:
    """A short phrase looped back to back several times is the classic
    "decoder got stuck" artifact -- real speech does not do this."""
    words = text.lower().split()
    for n in (1, 2, 3, 4):
        needed = n * min_repeats
        if len(words) < needed:
            continue
        for start in range(len(words) - needed + 1):
            phrase = words[start : start + n]
            if all(
                words[start + k * n : start + (k + 1) * n] == phrase
                for k in range(min_repeats)
            ):
                return True
    return False


def lexical_word_ratio(text: str) -> float:
    """Fraction of tokens that look like actual words.

    "Ct-4, M.D.S." tokenises to ct/m/d/s -- no vowels, all initials -> 0.0.
    "Uday, tell me about yourself." -> 1.0. This is what separates a mangled
    noise transcript from a terse but real one, without needing a dictionary
    or caring about length.
    """
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return 0.0

    real = 0
    for token in tokens:
        lowered = token.lower()
        if lowered in _VOWELLESS_REAL_WORDS:
            real += 1
            continue
        if not _LATIN_TOKEN_RE.match(lowered):
            # Written in some other script. The vowel/length heuristic below
            # cannot judge it, and the failure mode it defends against -- a
            # Whisper decoder emitting consonant clusters like "Ct.js." on
            # silence -- is a property of its Latin output, not of Devanagari
            # or CJK. Treat the token as a real word and let the other two
            # signals (voiced duration and decoder confidence) do the work.
            real += 1
            continue
        if len(lowered) >= 2 and _VOWEL_RE.search(lowered):
            real += 1
    return real / len(tokens)


def is_silence_phrase(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .!,")
    return normalized in _SILENCE_HALLUCINATIONS or not normalized


def assess_transcript(
    text: str,
    *,
    confidence: float = 1.0,
    confidence_known: bool = False,
    speech_seconds: float | None = None,
    min_lexical_ratio: float = 0.5,
    min_confidence: float = 0.35,
    silence_phrase_speech_seconds: float = 1.0,
) -> QualityVerdict:
    """Should this transcript be allowed into the question pipeline?

    `speech_seconds` is the VOICED duration measured by the VAD, not the
    segment length. `confidence_known` distinguishes "the provider told us
    0.9" from "we have no confidence signal and defaulted to 1.0" -- without
    it, a provider that reports nothing would look maximally trustworthy on
    every hallucination, which is exactly the bug this pipeline had.
    """
    stripped = text.strip()
    if not stripped:
        return QualityVerdict(False, "empty")

    if looks_like_stt_hallucination(stripped):
        return QualityVerdict(False, "looping_repetition")

    ratio = lexical_word_ratio(stripped)
    if ratio == 0.0:
        return QualityVerdict(False, "no_real_words")

    if ratio < min_lexical_ratio:
        return QualityVerdict(False, "mostly_non_words")

    # A stock silence phrase is only credible when there was enough voiced
    # audio to have actually said something.
    if is_silence_phrase(stripped):
        if speech_seconds is not None and speech_seconds < silence_phrase_speech_seconds:
            return QualityVerdict(False, "silence_hallucination")
        if confidence_known and confidence < min_confidence:
            return QualityVerdict(False, "silence_hallucination")

    if confidence_known and confidence < min_confidence:
        return QualityVerdict(False, "low_confidence")

    return QualityVerdict(True, "ok")
