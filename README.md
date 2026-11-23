# Fine-tune a small LLM to read pathology reports

A practice project. It teaches a small open model one narrow job:
read a pathology report, reply with clean JSON.

**In:**

```
Prostate, radical prostatectomy: Acinar adenocarcinoma, grade 3,
measuring 8.2 cm in greatest dimension. Tumor is present at the inked margin.
```

**Out:**

```json
{"organ": "prostate", "diagnosis": "acinar adenocarcinoma", "grade": 3, "size_mm": 82, "margins": "positive"}
```

The model has to learn our rules: "G3" and "poorly differentiated" both mean
grade `3`, "8.2 cm" means `82` mm, "free of tumor" means `negative`.

All reports are fake. `make_data.py` invents them. No patient data.

## How it works

We do not retrain the model. The model stays frozen. We train a small add-on
called a LoRA adapter (about 0.4% of the model's size). If the adapter is bad,
delete it. The base model is untouched.

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (in Ollama: `qwen2.5:0.5b`)
- **Data:** 270 examples to train on, 30 held back to test with

## Files

| File | What it does |
|---|---|
| `common.py` | Settings: model name, the prompt |
| `make_data.py` | Step 1: makes the fake reports |
| `data/train.jsonl`, `data/test.jsonl` | The data (already made) |
| `train.py` | Step 2: trains the adapter |
| `evaluate.py` | Step 3: before/after scores |
| `merge.py` | Step 4: glues base + adapter into one model |
| `Modelfile` | Step 5: tells Ollama how to load it |

## Run it

Needs [uv](https://docs.astral.sh/uv/). It installs the right Python and the
packages for you. A GPU makes it fast. It is used automatically.
`train.py` prints `training on: <GPU name>`. If it prints `CPU (no GPU found)`,
PyTorch cannot see the GPU: check `nvidia-smi` and the torch version.

```bash
uv sync                  # installs everything into .venv/

uv run make_data.py      # optional, the data is already in data/
uv run train.py          # makes adapter/
uv run evaluate.py       # prints before/after
```

`evaluate.py` is the part to show your boss. It prints how many of the 30
test reports the base model gets right, then how many base + adapter gets right.

## Put it in Ollama

Ollama cannot read this model's raw files. It needs one `.gguf` file.
llama.cpp has the converter. Its Python packages clash with ours, so uv
runs it in a separate throwaway environment.

```bash
uv run merge.py          # makes merged/

git clone --depth 1 https://github.com/ggml-org/llama.cpp
uv run --no-project --python 3.12 \
  --with-requirements llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  llama.cpp/convert_hf_to_gguf.py merged --outfile path-report-lora.gguf --outtype f16

ollama create path-report-lora -f Modelfile
ollama run path-report-lora "Colon, hemicolectomy: Adenocarcinoma, G2, measuring 31 mm. Margins are free of tumor."
```

## What was tested

- Tested on an Intel Mac, CPU only: `make_data.py`, `train.py` (3 steps only),
  `evaluate.py` (3 reports only), `merge.py`.
- Not tested: a full training run, anything on a GPU, the GGUF convert step,
  and the `Modelfile`.

## Change things

- Different model: edit `BASE_MODEL` in `common.py`. With a GPU, try
  `Qwen/Qwen2.5-1.5B-Instruct`.
- Train longer: edit `num_train_epochs` in `train.py`.
- Adapter not clearly better? Fix the data first. That is the usual cause.

<br>
