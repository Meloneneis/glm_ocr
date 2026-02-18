"""
Token probabilities from GLM-OCR generation — no library source changes.

Uses the standard Transformers API: model.generate(..., output_scores=True,
return_dict_in_generate=True). The returned .scores are logits (one tensor per
generated token); we apply softmax and record the probability of the chosen
token at each step.
"""
from __future__ import annotations

import torch
from typing import List, Tuple, Optional

# Type alias for ordered (token_str, probability) per sequence
TokenProbPairs = List[Tuple[str, float]]

# Optional: if you want to truncate at EOS for each sequence
# from transformers import AutoProcessor


def generate_with_token_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 1024,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **generate_kwargs,
) -> Tuple[torch.LongTensor, List[List[float]]]:
    """
    Call model.generate with output_scores=True and return sequences plus
    per-token probabilities (probability of the chosen token at each step).

    Args:
        model: The generation model (e.g. GlmOcrForConditionalGeneration).
        input_ids: (batch_size, seq_len).
        attention_mask, max_new_tokens, pad_token_id, eos_token_id: passed to generate.
        **generate_kwargs: Any other arguments for generate (e.g. do_sample, top_p).

    Returns:
        sequences: (batch_size, seq_len) generated token ids (same as standard generate).
        token_probs_per_batch: list of length batch_size; each element is a list of
            floats, the probability of the chosen token at each generated position.
            Positions after EOS (if any) are excluded so lengths can differ per sequence.
    """
    input_len = input_ids.shape[1]
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        **generate_kwargs,
    )
    sequences = out.sequences  # (batch_size, input_len + n_steps)
    scores = out.scores       # tuple of (batch_size, vocab_size), length n_steps

    if not scores:
        n = sequences.shape[0]
        return sequences, [[] for _ in range(n)]

    generated = sequences[:, input_len:]  # (batch_size, n_steps)
    batch_size = sequences.shape[0]
    n_steps = generated.shape[1]

    token_probs_per_batch: List[List[float]] = []
    for b in range(batch_size):
        token_probs: List[float] = []
        for t in range(n_steps):
            token_id = generated[b, t].item()
            if eos_token_id is not None and token_id == eos_token_id:
                # Stop recording after EOS
                break
            logits = scores[t][b]  # (vocab_size,)
            probs = torch.softmax(logits.float(), dim=-1)
            token_probs.append(probs[token_id].item())
        token_probs_per_batch.append(token_probs)

    return sequences, token_probs_per_batch


def _scores_to_token_probs(
    sequences: torch.Tensor,
    scores: tuple,
    input_len: int,
    eos_token_id: Optional[int],
) -> List[List[float]]:
    """Convert generate() scores to per-sequence lists of chosen-token probabilities."""
    if not scores:
        return [[] for _ in range(sequences.shape[0])]
    generated = sequences[:, input_len:]
    batch_size = sequences.shape[0]
    n_steps = generated.shape[1]
    token_probs_per_batch: List[List[float]] = []
    for b in range(batch_size):
        token_probs: List[float] = []
        for t in range(n_steps):
            token_id = generated[b, t].item()
            if eos_token_id is not None and token_id == eos_token_id:
                break
            logits = scores[t][b]
            probs = torch.softmax(logits.float(), dim=-1)
            token_probs.append(probs[token_id].item())
        token_probs_per_batch.append(token_probs)
    return token_probs_per_batch


def _token_probs_to_ordered_pairs(
    generated_token_ids: torch.Tensor,
    token_probs: List[float],
    tokenizer,
) -> TokenProbPairs:
    """Build an ordered list of (decoded_token_str, probability), excluding special tokens."""
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    pairs: TokenProbPairs = []
    for tid, prob in zip(generated_token_ids.tolist(), token_probs):
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        if token_str in special_tokens:
            continue
        pairs.append((token_str, prob))
    return pairs


def run_ocr_batch_with_probs(
    processor,
    model,
    images: list,
    prompt: str = "Text Recognition:",
    max_new_tokens: int = 1024,
) -> Tuple[List[str], List[TokenProbPairs]]:
    """
    Run GLM-OCR on a batch of PIL images and return decoded texts plus
    ordered per-token (token_str, probability) pairs for each image.

    Returns:
        texts: list of cleaned OCR strings (one per image).
        token_probs_per_image: list of ordered lists; token_probs_per_image[i] is
            a list of (token_str, probability) tuples for the i-th image, in
            generation order (every output token with its probability).
    """
    if not images:
        return [], []

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for img in images
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Use same generate() args as non-probs path so output is identical. Only add what we need for scores.
    # Do NOT pass pad_token_id/eos_token_id here—they change batched generation (early stop + padding)
    # and cause decoded text to include padding/garbage. We only need eos_token_id for post-processing.
    eos_token_id = getattr(model.config, "eos_token_id", None)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = out.sequences
    scores = out.scores or []
    input_len = inputs["input_ids"].shape[1]
    token_probs_per_batch = _scores_to_token_probs(sequences, scores, input_len, eos_token_id)

    # Build ordered (token_str, probability) list per sequence
    generated = sequences[:, input_len:]
    tokenizer = processor.tokenizer
    token_probs_ordered: List[TokenProbPairs] = [
        _token_probs_to_ordered_pairs(generated[b], token_probs_per_batch[b], tokenizer)
        for b in range(sequences.shape[0])
    ]

    decoded = processor.batch_decode(sequences, skip_special_tokens=True)
    # TokenizersBackend does not strip <think>/</think>/<|image|> despite skip_special_tokens=True; strip in post.
    def _clean(s: str) -> str:
        if prompt in s:
            s = s.split(prompt)[-1]
        return s.replace("<think>", "").replace("</think>", "").replace("<|image|>", "").strip() or ""
    texts = [_clean(s) for s in decoded]

    return texts, token_probs_ordered
