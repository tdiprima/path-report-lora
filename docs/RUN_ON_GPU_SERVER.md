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

Check the version. It must be 3.10, 3.11 or 3.12.

```bash
python3 --version
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The install downloads about 2.5 GB. It takes a few minutes.

## 5. Check that PyTorch sees the GPU

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

It must print `True`. If it prints `False`, stop and see "If something goes
wrong" below. Training on `False` still works, but slowly, on the CPU.

## 6. Train

```bash
python train.py
```

Near the top it prints `training on: <GPU name>`. Then a progress bar, with
the loss printed every 10 steps. The loss should go down. When it finishes
it prints `adapter saved to adapter/`. Expect a minute or two.

To use a specific GPU, for example number 1:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

## 7. Get the before/after scores

```bash
python evaluate.py | tee results.txt
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
need the admin. If `nvidia-smi` works, the GPU is probably too new for the
pinned PyTorch. Open `requirements.txt`, raise only the `torch==` line to a
newer version, and run `pip install -r requirements.txt` again.

**`python3 --version` shows 3.13 or newer, or 3.9 or older.**
Try `python3.12 --version` or `python3.11 --version`. If one exists, use it
in place of `python3` in step 4. Some servers need `module avail python` and
`module load ...` first.

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
