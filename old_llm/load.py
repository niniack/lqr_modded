# Loads model from huggingface

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_hf(model_name, dtype="auto", device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map
    )
    return tokenizer, model