"""Step 1: make fake pathology reports and the JSON answer for each one.

Everything here is invented. No patient data.
Writes data/train.jsonl (270 examples) and data/test.jsonl (30 examples).
"""

import json
import random
from pathlib import Path

random.seed(42)

# organ -> (specimen types, diagnoses)
ORGANS = {
    "breast": (["lumpectomy", "mastectomy"], ["invasive ductal carcinoma", "invasive lobular carcinoma"]),
    "colon": (["hemicolectomy", "segmental resection"], ["adenocarcinoma", "mucinous adenocarcinoma"]),
    "lung": (["lobectomy", "wedge resection"], ["adenocarcinoma", "squamous cell carcinoma"]),
    "prostate": (["radical prostatectomy"], ["acinar adenocarcinoma"]),
    "kidney": (["partial nephrectomy", "radical nephrectomy"], ["clear cell renal cell carcinoma", "papillary renal cell carcinoma"]),
    "skin": (["wide excision", "excisional biopsy"], ["squamous cell carcinoma", "basal cell carcinoma"]),
    "stomach": (["partial gastrectomy"], ["adenocarcinoma"]),
    "bladder": (["cystectomy"], ["urothelial carcinoma"]),
}

# Many ways a pathologist writes the same grade. The model must learn they mean 1, 2 or 3.
GRADES = {
    1: ["well differentiated", "grade 1", "G1", "low grade (1 of 3)"],
    2: ["moderately differentiated", "grade 2", "G2", "intermediate grade (2 of 3)"],
    3: ["poorly differentiated", "grade 3", "G3", "high grade (3 of 3)"],
}

MARGINS = {
    "negative": [
        "Margins are free of tumor.",
        "All surgical margins are uninvolved.",
        "No tumor at the inked margin.",
        "Resection margins negative for carcinoma.",
    ],
    "positive": [
        "Tumor is present at the inked margin.",
        "The deep margin is involved by carcinoma.",
        "Margins positive for tumor.",
        "Carcinoma extends to the resection margin.",
    ],
}

TEMPLATES = [
    "SPECIMEN: {organ_cap}, {specimen}.\nFINAL DIAGNOSIS: {diagnosis_cap}, {grade}.\nTumor size: {size}.\n{margins}",
    "{organ_cap}, {specimen}: {diagnosis_cap}, {grade}, measuring {size} in greatest dimension. {margins}",
    "Received: {specimen} specimen of {organ}. Sections show {diagnosis}, {grade}. The tumor measures {size}. {margins}",
    "DIAGNOSIS\n{organ_cap} ({specimen}):\n- {diagnosis_cap}\n- Histologic grade: {grade}\n- Greatest dimension: {size}\n- {margins}",
    "A {size} mass is identified in the {organ} {specimen}. Microscopy: {diagnosis}, {grade}. {margins}",
]


def make_example():
    organ = random.choice(list(ORGANS))
    specimens, diagnoses = ORGANS[organ]
    diagnosis = random.choice(diagnoses)
    grade = random.choice([1, 2, 3])
    size_mm = random.randint(4, 95)
    margins = random.choice(["negative", "positive"])

    # Reports give size in cm or mm. The answer is always mm.
    size_text = f"{size_mm / 10:g} cm" if random.random() < 0.6 else f"{size_mm} mm"

    report = random.choice(TEMPLATES).format(
        organ=organ,
        organ_cap=organ.capitalize(),
        specimen=random.choice(specimens),
        diagnosis=diagnosis,
        diagnosis_cap=diagnosis.capitalize(),
        grade=random.choice(GRADES[grade]),
        size=size_text,
        margins=random.choice(MARGINS[margins]),
    )
    answer = {"organ": organ, "diagnosis": diagnosis, "grade": grade, "size_mm": size_mm, "margins": margins}
    return {"input": report, "output": json.dumps(answer)}


def main():
    examples = {}
    while len(examples) < 300:  # dict keyed by report text = no duplicates
        example = make_example()
        examples[example["input"]] = example
    examples = list(examples.values())

    Path("data").mkdir(exist_ok=True)
    splits = {"data/train.jsonl": examples[:270], "data/test.jsonl": examples[270:]}  # hold back 10%
    for path, rows in splits.items():
        Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"wrote {len(rows)} examples to {path}")


if __name__ == "__main__":
    main()
