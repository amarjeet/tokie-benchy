from __future__ import annotations

import random

_WORDS = (
    "river mountain lantern harbour quiet signal copper meadow orbit thread canvas "
    "ember valley compass willow granite summer archive velvet kernel horizon pulse "
    "marble cedar prism tundra saffron beacon glacier anchor nectar fable drift "
    "ledger quartz timber cobalt lattice meridian tallow ripple sundial vessel"
).split()

_INSTRUCTION = "Continue this passage at length. Never stop early or summarise.\n\n"

# Rough ratio for English-like filler on common BPE tokenizers.
_TOKENS_PER_WORD = 1.35
# Measured fixed cost: instruction + chat template + any server-injected system prompt.
# Model/proxy dependent; the server-reported prompt_tokens is what gets recorded.
_OVERHEAD_TOKENS = 64


def build_prompt(target_tokens: int, seed: int) -> str:
    """Build a deterministic prompt of roughly `target_tokens` tokens.

    Each seed yields distinct text so prefix caching on the server does not skew TTFT.
    """
    rng = random.Random(seed)
    n_words = max(8, int((target_tokens - _OVERHEAD_TOKENS) / _TOKENS_PER_WORD))
    words = []
    for i in range(n_words):
        word = rng.choice(_WORDS)
        if i % 11 == 0:
            word = word.capitalize()
        elif i % 11 == 10:
            word += "."
        words.append(word)
    return _INSTRUCTION + " ".join(words)
