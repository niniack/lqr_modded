from __future__ import annotations

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "ibm-granite/granite-guardian-4.1-8b"

COHERENCE = (
    "Judge coherence only. Score 'yes' when the visible text is readable and "
    "a reader can follow its train of thought. Score 'no' only when the text "
    "is dominated by garbled language, disconnected ideas, or repetition "
    "loops with little meaningful progression. Evaluate it as a continuation "
    "of the preceding prompt: starting or stopping mid-sentence, a cut-off "
    "word, or an unfinished story does not justify 'no'. Minor grammar "
    "issues, awkward style, and mild repetition are acceptable. Ignore "
    "factual accuracy, topic relevance, and whether the response fully "
    "answers the prompt. Scoring polarity: coherent = 'yes'; incoherent = 'no'."
)


def concept_criterion(concept):
    return (
        f"The response meaningfully includes or discusses the idea or topic {concept!r}. "
        "Count clear paraphrases as well as direct references; an unrelated or "
        "incidental mention alone does not count."
    )


_INSTRUCTION = (
    "<guardian><no-think>As a judge agent, assess whether the provided text meets the "
    "given judging criteria using all available information, including "
    "conversations, documents, and tools. Provide your score immediately "
    "without explanation. Output empty <think>\n</think> tags followed by "
    "your score in <score></score> tags.\n\n"
    "### Criteria: {criterion}\n\n"
    "### Scoring Schema: If the last assistant's text meets the criteria, "
    "return 'yes'; otherwise, return 'no'."
)


def build_judge(device_map="cuda:0", batch_size=32):
    """Load Granite once; return judge(criterion, prompt, responses, *, desc="Judging").

    Score yes/no after the requested empty thinking/score prefix in one forward pass,
    rather than autoregressively generating the fixed XML wrapper. Duplicate responses
    are evaluated once per call; length-sorted batches minimize padding. Results retain
    the input order. Progress counts unique responses, not generated wrapper tokens.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, device_map=device_map,
                                                 dtype=torch.bfloat16).eval()
    prefix = tokenizer.encode("<think>\n</think>\n<score>", add_special_tokens=False)
    yes, no = (tokenizer.encode(word, add_special_tokens=False) for word in (" yes", " no"))
    if len(yes) != 1 or len(no) != 1:
        raise RuntimeError("Granite yes/no labels must each be a single token")
    yes, no = yes[0], no[0]

    @torch.inference_mode()
    def judge(criterion, prompt, responses, *, desc="Judging"):
        responses = list(responses)
        unique = list(dict.fromkeys(responses))
        if not unique:
            return []
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response},
             {"role": "user", "content": _INSTRUCTION.format(criterion=criterion)}],
            tokenize=False, add_generation_prompt=True) for response in unique]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        encoded = [ids + prefix for ids in encoded]
        order = sorted(range(len(unique)), key=lambda i: len(encoded[i]))
        verdicts = {}
        with tqdm(total=len(unique), desc=desc, unit="response") as progress:
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                inputs = tokenizer.pad({"input_ids": [encoded[i] for i in indices]},
                                       padding=True, return_tensors="pt").to(model.device)
                positions = inputs["attention_mask"].long().cumsum(-1) - 1
                positions.masked_fill_(inputs["attention_mask"] == 0, 1)
                logits = model(**inputs, position_ids=positions, use_cache=False,
                               logits_to_keep=1).logits[:, -1]
                labels = logits.argmax(-1).tolist()
                for i, label in zip(indices, labels):
                    if label not in (yes, no):
                        raise RuntimeError(f"Unexpected judge verdict token: {tokenizer.decode([label])!r}")
                    verdicts[unique[i]] = label == yes
                progress.update(len(indices))
        return [verdicts[response] for response in responses]

    return judge


def repetition(text, n=4):
    """Fraction of word n-grams that repeat an earlier n-gram in the same text."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0
