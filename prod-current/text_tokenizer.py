# SPDX-License-Identifier: Apache-2.0
"""Text-prompt builder for HiggsMultimodalQwen3 TTS.

Assembles, depending on ref-audio + ref-text presence:

- voice-clone + transcript: ``<|tts|> <|ref_text|> tok(ref) <|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>``
- voice-clone, no transcript: ``<|tts|> <|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>``
- zero-shot: ``<|tts|> <|text|> tok(text) <|audio|>``

Multi-turn longform (``context_turns`` set) interleaves completed
``<|text|> tok(t_i) <|audio|> [-100]×A_i`` pairs between the ref block and the
current ``<|text|>``, so the model grounds the next chunk on the prosody it
just produced::

    <|tts|> <|ref_text|> tok(ref) <|ref_audio|> [-100]×N
            <|text|> tok(t_1) <|audio|> [-100]×A_1     # prior turn 1
            <|text|> tok(t_2) <|audio|> [-100]×A_2     # prior turn 2
            <|text|> tok(text) <|audio|>               # current → generated here

``<|tts|>`` selects task mode (vs ASR); missing it yields fluent-but-wrong
output. ``-100`` placeholders are spliced by :class:`HiggsFusedMultiTextEmbedding`
at runtime; the embedding stage consumes ``reference_codes_delayed`` by summing
*all* ``-100`` positions in the request span, so for multi-turn that tensor MUST
be ``concat(ref_delayed, A_1_delayed, A_2_delayed, …)`` in placeholder order.
Each ``num_ref_tokens`` / ``A_i`` must match the *delayed* code row count of its
segment (``T + num_codebooks - 1``).
"""

from __future__ import annotations

from typing import Any

# Matches Higgs ``audio_token_id`` and transformers' ``IGNORE_INDEX`` convention.
AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS: tuple[str, ...] = (
    "<|tts|>",
    "<|ref_audio|>",
    "<|text|>",
    "<|audio|>",
)


class HiggsTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.tts_id: int = vocab["<|tts|>"]
        self.ref_audio_id: int = vocab["<|ref_audio|>"]
        self.text_id: int = vocab["<|text|>"]
        self.audio_id: int = vocab["<|audio|>"]
        # Newer ckpts only; older ckpts fall back to audio-only voice-cloning.
        self.ref_text_id: int | None = vocab.get("<|ref_text|>")

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def build_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
        context_turns: list[tuple[str, int]] | None = None,
    ) -> list[int]:
        """``num_ref_tokens=0`` → zero-shot; non-zero must match delayed row count.

        ``context_turns`` is an ordered list of ``(text_i, num_audio_tokens_i)``
        prior turns (each ``num_audio_tokens_i`` = that chunk's *delayed* code row
        count). They are emitted as ``<|text|> tok(text_i) <|audio|> [-100]×A_i``
        between the ref block and the current text. The caller MUST set
        ``reference_codes_delayed`` to ``concat(ref, A_1, A_2, …)`` so the runtime
        placeholder-paste consumes the right codes for each block in order.
        """
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        ids: list[int] = [self.tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        for turn_idx, (turn_text, turn_audio_tokens) in enumerate(context_turns or []):
            if turn_audio_tokens <= 0:
                raise ValueError(
                    f"context_turns[{turn_idx}] audio tokens must be > 0, "
                    f"got {turn_audio_tokens}"
                )
            ids.append(self.text_id)
            ids.extend(self._tok.encode(turn_text, add_special_tokens=False))
            ids.append(self.audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * turn_audio_tokens)
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids


__all__ = ["AUDIO_PLACEHOLDER_ID", "HiggsTokenizerAdapter"]
