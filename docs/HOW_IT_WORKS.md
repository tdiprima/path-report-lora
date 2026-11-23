# How This System Works

## 1. What It Does

It teaches a small AI model one narrow skill: read a pathology report and
reply with a tidy JSON summary.

**Input:** a short pathology report in plain English.

**What the system does:** makes practice reports with known answers, uses
them to train a small add-on for an existing model, then measures whether the
add-on helped.

**Final output:** two things.

1. A scorecard: how the model did before training versus after.
2. A trained model you can run in Ollama.

The model already exists (Qwen2.5, 0.5B size, made by Alibaba, free to
download). This project does not build a model. It adjusts one.

## 2. The Whole Process in 60 Seconds

1. Invent 300 fake pathology reports. Each comes with its correct JSON answer.
2. Set 30 aside. The model never sees these during training.
3. Download the base model and freeze it.
4. Attach a small trainable add-on (a "LoRA adapter") to the frozen model.
5. Show the model the 270 training reports. Each time its answer is wrong,
   nudge the add-on so the next answer is less wrong.
6. Save the add-on.
7. Give the 30 held-back reports to the plain model and score its answers.
8. Give the same 30 to the model plus add-on and score again. Compare.
9. Glue the add-on into the model, convert to one file, load it into Ollama.

```
 make fake reports
        |
        +--> 270 for training ----> TRAIN ----> adapter (the small add-on)
        |                             ^              |
        |                     frozen base model      |
        |                                            v
        +--> 30 held back  ---------------------> SCORE: base alone
                                                  SCORE: base + adapter
                                                         |
                                                         v
                                                  before / after table

 adapter + base model --> MERGE --> convert to .gguf --> Ollama
```

## 3. The Major Jobs

### Job 1: Invent the practice data

**What happens.** A script writes fake pathology reports and, for each one,
the JSON answer a perfect reader would give.

**Why.** Training needs examples of the question together with the right
answer. Real reports contain patient data, so the system makes its own.

**How.** For each report it rolls dice for five facts:

- organ (breast, colon, lung, prostate, kidney, skin, stomach, bladder)
- diagnosis (picked from a list that fits that organ)
- grade (1, 2 or 3)
- tumor size (4 to 95 mm)
- margins (negative or positive)

Then it drops those facts into one of five sentence layouts. This is where
the learning challenge comes from: the report and the answer deliberately
say the same thing in different ways.

| Fact | Ways the report may say it | What the answer must say |
|---|---|---|
| Grade 2 | "moderately differentiated", "grade 2", "G2", "intermediate grade (2 of 3)" | `2` |
| Size 82 mm | "8.2 cm" (60% of the time) or "82 mm" | `82` |
| Negative margins | "Margins are free of tumor.", "No tumor at the inked margin.", and two more | `"negative"` |

So the model cannot copy words across. It has to translate.

**Rules.** The dice use a fixed seed, so every run produces the same 300
reports. Duplicate reports are thrown away.

**What comes out.** Two files of input/output pairs: 270 for training, 30 for
testing. One pair per line (the JSONL format).

**Where it goes.** The 270 go to Job 3. The 30 go to Job 4.

### Job 2: Wrap each report in a conversation

**What happens.** A report is never sent to the model bare. It is packed into
a three-part chat:

- **system:** the instructions ("Extract fields... reply with JSON only, with
  exactly these keys...")
- **user:** the report
- **assistant:** the JSON answer (present in training, left blank when asking)

**Why.** Chat models are built to read conversations in this shape. And the
wording must be identical everywhere. A model trained with one set of
instructions and tested with another gets worse scores for no good reason.

**How.** One shared helper builds this chat. Training, scoring and Ollama all
use the same instruction text.

**What comes out.** A chat ready to be turned into numbers.

### Job 3: Train the add-on

This is the core of the system.

**What happens.** The model practises on the 270 reports and a small add-on
learns the skill.

**Why an add-on, and not the model itself.** The model has about 496 million
numbers inside it ("weights"). Changing all of them is slow, needs lots of
memory, and can wreck the model. So the system freezes every one of them and
attaches a small set of new numbers beside the model's attention layers (the
parts that decide which words to focus on). Only those new numbers are
trained. That is LoRA.

Here the add-on is about 2.2 million numbers, which is 0.44% of the model.
If training goes badly, delete the add-on. The original is untouched.

**How.**

1. Each chat is turned into tokens (small word pieces, each with an ID
   number), because models read numbers, not text.
2. The model reads the chat and predicts the next token, over and over.
3. Its predictions are compared with the real answer. The gap is called the
   loss. Big loss means bad guesses.
4. The add-on's numbers are nudged in the direction that shrinks the loss.
5. Repeat for every report, twice through the whole set (2 "epochs").

**The important decision: grade only the answer.** The model is not graded on
predicting the instructions or the report, only on the JSON. The prompt part
is marked "ignore" (the code uses the label `-100` for this). Otherwise the
model would spend effort learning to recite reports, which nobody wants.

**Other settings that matter.**

- One report at a time, with the add-on updated after every 4 reports. That
  is about 135 updates in total.
- The add-on's size setting is `r=16`. Bigger means more capacity to learn
  and slower training.
- A GPU is used if the machine has one. Otherwise the CPU, slowly.

**What comes out.** A small folder holding only the add-on.

**Where it goes.** To Job 4 for scoring and Job 5 for packaging.

### Job 4: Score before and after

**What happens.** The 30 held-back reports are given to the model twice:
first with no add-on, then with it.

**Why.** Without a comparison you cannot tell whether training helped. And
the test reports must be ones the model never trained on, or it could score
well just by remembering.

**How each reply is judged.**

1. The model writes a reply (up to 80 tokens, always picking its single most
   likely next token, so the result is repeatable).
2. The reply is parsed as JSON. If that fails, every field counts as wrong.
   A chatty reply like "Sure! Here is the JSON:" fails here.
3. Each of the five fields is compared with the truth. The match is strict:
   `"3"` in quotes is not `3`, and `8.2` is not `82`.
4. The whole answer counts as right only if it equals the truth exactly.
   One wrong field, or one extra field, fails it.

**What comes out.** Two score tables (valid JSON count, a count per field,
whole-answer count), plus one report shown with the correct answer, the
"before" reply and the "after" reply side by side.

**Where it goes.** To a human. This is the evidence that the fine-tune worked
or did not.

### Job 5: Package it for Ollama

**What happens.** The add-on is folded into the base model, and the result is
converted into a single file Ollama can run.

**Why.** During training, the model and add-on are two separate things.
Ollama wants one model in one file, in a format called GGUF.

**How.**

1. **Merge.** The add-on's numbers are added into the base model's numbers.
   The result is an ordinary model that behaves like base plus add-on.
2. **Convert.** A converter script from the llama.cpp project turns the
   merged folder into one `.gguf` file. This step is run by hand and is not
   part of this project's code.
3. **Register.** A small recipe file (the `Modelfile`) tells Ollama where the
   file is, which instructions to always send, and how to lay out the chat.
   It also sets temperature to 0, so the model does not answer randomly.

**What comes out.** A model named `path-report-lora` inside Ollama.

**Status.** The merge step has been run. The convert and register steps have
not been tested.

## 4. How the Pieces Work Together

Each job hands one thing to the next.

- **Data to training.** The 270 pairs are the only thing the add-on learns
  from. Whatever patterns are in the data are the patterns it learns. If the
  result is poor, the data is the first suspect.
- **Data to scoring.** The 30 held-back pairs are the answer key.
- **The shared instructions.** The same system prompt is used in training,
  scoring and Ollama. This is the thread holding the jobs together. Change it
  in one place only and the trained model sees wording it never practised on.
- **Training to scoring.** Scoring loads the frozen base model, tests it,
  then loads the add-on onto that same model and tests again. Same model,
  same reports, one difference. That is what makes the comparison fair.
- **Training to packaging.** The merge needs the same base model the add-on
  was trained on. An add-on only fits the model it was made for.

One settings file names the base model, the folders and the instructions, so
every job agrees on them.

## 5. Important External Pieces

- **Qwen2.5-0.5B-Instruct.** The base model. Small, free, open. Downloaded
  from Hugging Face on first run. Chosen small so it can train on modest
  hardware.
- **Hugging Face `transformers`.** Loads the model, turns text into tokens,
  and runs the training loop, so this project does not write one by hand.
- **`peft`.** Creates the LoRA add-on, freezes the base model, saves the
  add-on, and later merges it.
- **PyTorch.** The math engine underneath both. Decides CPU or GPU.
- **llama.cpp.** Only its converter script is used, to make the `.gguf` file.
- **Ollama.** Runs the finished model on your own machine.

Nothing is sent to an outside service. The only network use is downloading
the model and the libraries.

## 6. End-to-End Example

Follow one report through the system.

**Job 1** rolls: prostate, acinar adenocarcinoma, grade 3, 82 mm, positive
margins. It picks a layout, writes the size as cm, and produces this pair:

```
input:  Prostate, radical prostatectomy: Acinar adenocarcinoma, grade 3,
        measuring 8.2 cm in greatest dimension. Tumor is present at the
        inked margin.

output: {"organ": "prostate", "diagnosis": "acinar adenocarcinoma",
         "grade": 3, "size_mm": 82, "margins": "positive"}
```

This one lands in the held-back 30, so training never sees it.

**Job 3** trains on the other 270. Those include other reports where cm
became mm and "inked margin" meant positive, so the add-on can learn the
rules without seeing this exact report.

**Job 4, before.** The report is wrapped in the chat (Job 2) and given to the
plain model. In a quick trial it replied:

```json
{
  "organ": "prostate",
  "diagnosis": "radical prostatectomy",
  "grade": "3",
  "size_mm": "8.2",
  "margins": "positive"
}
```

It is valid JSON, so it passes the first check. Then: organ right. Diagnosis
wrong (it gave the surgery, not the disease). Grade wrong (text `"3"`, not
the number `3`). Size wrong (still in cm, and text). Margins right. Two of
five fields. Whole answer: wrong.

**Job 4, after.** Same report, model plus add-on. The goal is a reply equal
to the answer key, character for character in meaning. In that same quick
trial, where the add-on had only 3 updates of practice instead of the full
135, the reply already had the right shape and got four fields right, but
gave `820` for the size. A full training run has not been done yet, so the
real "after" score is not known.

**Job 5.** Once merged, converted and loaded, the same report typed into
`ollama run path-report-lora` should give the JSON back directly.

## 7. The Simplest Mental Model

It takes a model that already reads English, makes up a few hundred practice
questions with known answers, and uses most of them to train a small
detachable add-on while the model itself stays frozen. It then uses the
leftover questions as an exam, taken once without the add-on and once with
it, to show what changed. Finally it welds the add-on into the model and
packs it into one file so Ollama can run it.

Fake reports in. A scorecard and a specialised model out.
