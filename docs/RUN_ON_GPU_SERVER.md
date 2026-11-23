# Run it on a GPU server

These steps have not been run on a GPU server yet. They assume the server is a Linux
server with an NVIDIA GPU, internet access, and that you log in with plain
`ssh`. If jobs there must go through a scheduler such as Slurm, the commands
are the same but go inside a job script.

## 0. On your laptop: push your latest changes

The server gets the code from GitHub, so anything not pushed will be missing.

```bash
git status
```

If it lists changed files, commit and push them first.

## 1. Log in

```bash
ssh your-gpu-server
```

## 2. Check the GPU

```bash
nvidia-smi
```

You should see a table with at least one GPU. Look at the "Memory-Usage"
column. If a GPU is nearly full, someone else is using it. Note the number of
a free one (0, 1, 2...). You need about 4 GB free.

## 3. Get the code

```bash
git clone https://github.com/tdiprima/path-report-lora
cd path-report-lora
```

## 4. Set up Python

This project uses `uv`. Check that the server has it:

```bash
uv --version
```

If that says "command not found", install it (no admin rights needed), then
log out and back in:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
uv sync
```

This downloads the right Python and all packages into `.venv/`. About 2.5 GB.
It takes a few minutes. You do not need to activate anything: `uv run` uses
that folder by itself.

## 5. Check that PyTorch sees the GPU

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

It must print `True`. If it prints `False`, stop and see "If something goes
wrong" below. Training on `False` still works, but slowly, on the CPU.

## 6. Train

```bash
uv run train.py
```

Near the top it prints `training on: <GPU name>`. Then a progress bar, with
the loss printed every 10 steps. The loss should go down. When it finishes
it prints `adapter saved to adapter/`. Expect a minute or two.

To use a specific GPU, for example number 1:

```bash
CUDA_VISIBLE_DEVICES=1 uv run train.py
```

## 7. Get the before/after scores

```bash
uv run evaluate.py | tee results.txt
```

This prints the scores for the base model, then for base + adapter, then one
example. `tee` also saves a copy in `results.txt`. That file is what to show
your boss.

To copy it to your laptop, run this on your laptop:

```bash
scp your-gpu-server:path-report-lora/results.txt .
```

## 8. Optional: load it into Ollama

Only if Ollama is installed on the server (`ollama --version`). Follow the
"Put it in Ollama" section of the main `README.md`. That part is untested.

## If something goes wrong

**`torch.cuda.is_available()` prints `False`.**
Run `nvidia-smi`. If that fails too, the server has a driver problem and you
need the admin. If `nvidia-smi` works, check which PyTorch got installed:

```bash
uv run python -c "import torch; print(torch.__version__)"
```

On an ARM server (`uname -m` prints `aarch64`) it must end in `+cu126`. If it
does not, your copy of the code is old: run `git pull`, then `uv sync` again.
On an Intel/AMD server, the GPU is probably too new for the pinned PyTorch.
Open `pyproject.toml`, raise only the first `torch==` line to a newer version,
and run `uv sync` again.

**`uv sync` cannot download Python or packages.**
The server may block outside internet. Ask the admin about a proxy.

**`CUDA out of memory`.**
Another job is using the GPU. Pick a free one with `CUDA_VISIBLE_DEVICES`
(step 6), or wait.

**The model download fails.**
The server may block outside internet. The model comes from huggingface.co on
the first run. Ask the admin about a proxy.

**"No space left on device".**
The model is cached in your home folder (`~/.cache/huggingface`, about 1 GB).
To put it somewhere roomier, set this before running:

```bash
export HF_HOME=/path/with/space/huggingface
```

**The adapter is not clearly better in step 7.**
Train longer: change `num_train_epochs=2` to `4` in `train.py` and rerun
steps 6 and 7.
