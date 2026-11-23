"""Settings shared by every script. Change things here, nowhere else."""

# Small on purpose: this trains on a laptop CPU. Same model in Ollama: qwen2.5:0.5b
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_DIR = "adapter"  # the small add-on that training produces
MERGED_DIR = "merged"  # base + adapter glued together, for Ollama

FIELDS = ["organ", "diagnosis", "grade", "size_mm", "margins"]

SYSTEM_PROMPT = (
    "Extract fields from the pathology report. Reply with JSON only, with exactly "
    "these keys: organ, diagnosis, grade (1, 2 or 3), size_mm (whole number), "
    'margins ("negative" or "positive").'
)


def chat(report, answer=None):
    """Turn one report (and optionally its answer) into chat messages."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": report},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
