"""Step 3: before/after. Run the held-back test reports through the base
model, then through base + adapter, and compare the scores."""

import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ADAPTER_DIR, BASE_MODEL, FIELDS, chat

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

with open("data/test.jsonl") as f:
    test_data = [json.loads(line) for line in f]


def ask(model, report):
    inputs = tokenizer.apply_chat_template(
        chat(report), add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def score(model, name):
    """Returns the model's raw replies. Prints how many it got right."""
    replies, valid_json, all_right = [], 0, 0
    field_right = dict.fromkeys(FIELDS, 0)
    for example in test_data:
        reply = ask(model, example["input"])
        replies.append(reply)
        truth = json.loads(example["output"])
        try:
            guess = json.loads(reply)
        except json.JSONDecodeError:
            continue  # not JSON = every field wrong
        if not isinstance(guess, dict):
            continue
        valid_json += 1
        for field in FIELDS:
            field_right[field] += guess.get(field) == truth[field]
        all_right += guess == truth

    n = len(test_data)
    print(f"\n=== {name} ===")
    print(f"valid JSON:      {valid_json}/{n}")
    for field in FIELDS:
        print(f"{field + ':':16} {field_right[field]}/{n}")
    print(f"whole answer:    {all_right}/{n}")
    return replies


model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
before = score(model, "BASE MODEL")

model = PeftModel.from_pretrained(model, ADAPTER_DIR)
after = score(model, "BASE + ADAPTER")

print("\n=== ONE EXAMPLE ===")
print(f"report:\n{test_data[0]['input']}\n")
print(f"correct: {test_data[0]['output']}")
print(f"before:  {before[0]}")
print(f"after:   {after[0]}")
