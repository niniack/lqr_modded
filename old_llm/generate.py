def generate_reply(model, tokenizer, prompt, *, max_new_tokens=32768,
                   enable_thinking=False):
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0, inputs.input_ids.shape[-1]:].tolist()

    # Look up the delimiter instead of hard-coding its token ID.
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if end_think_id is None or end_think_id == tokenizer.unk_token_id:
        split_at = 0
    else:
        try:
            split_at = len(output_ids) - output_ids[::-1].index(end_think_id)
        except ValueError:
            split_at = 0

    # thinking = tokenizer.decode(
    #     output_ids[:split_at], skip_special_tokens=True
    # ).strip("\n")
    
    answer = tokenizer.decode(
        output_ids[split_at:], skip_special_tokens=True
    ).strip("\n")

    return answer