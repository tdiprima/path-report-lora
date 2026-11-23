"""Step 4: glue base + adapter into one normal model, so Ollama can load it."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ADAPTER_DIR, BASE_MODEL, MERGED_DIR

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload()
model.save_pretrained(MERGED_DIR)
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(MERGED_DIR)
print(f"merged model saved to {MERGED_DIR}/")
