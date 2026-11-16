
import json
import random
from pathlib import Path
from collections import defaultdict

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "standardized"
    / "combined_metadata.jsonl"
)

OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "splits"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def get_group_id(record):
    dataset = str(record["dataset"]).lower()

    # SIDTD: connect fake documents to their original source image
    if dataset == "sidtd":
        source_image = record.get("source_image")

        if source_image and str(source_image).lower() != "none":
            original_document = Path(str(source_image)).stem
        else:
            original_document = record["document_id"]

        return (dataset, original_document)

    # Kriyam: keep all versions of the same document together
    return (dataset, record["document_id"])


def main():

    random.seed(RANDOM_SEED)

    # Load combined dataset
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    # Group records by their original document
    document_groups = defaultdict(list)

    for record in records:
        group_id = get_group_id(record)
        document_groups[group_id].append(record)

    # Shuffle document groups
    groups = list(document_groups.values())
    random.shuffle(groups)

    total_groups = len(groups)

    train_end = int(total_groups * TRAIN_RATIO)
    val_end = train_end + int(total_groups * VAL_RATIO)

    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]

    # Flatten groups into records
    train_records = [
        record for group in train_groups for record in group
    ]

    val_records = [
        record for group in val_groups for record in group
    ]

    test_records = [
        record for group in test_groups for record in group
    ]

    # Save split files
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.jsonl": train_records,
        "validation.jsonl": val_records,
        "test.jsonl": test_records
    }

    for filename, split_records in splits.items():

        output_file = OUTPUT_ROOT / filename

        with open(output_file, "w", encoding="utf-8") as file:
            for record in split_records:
                file.write(json.dumps(record) + "\n")

    # Display statistics
    print("\nDATASET SPLITTING COMPLETED!")
    print("--------------------------------")
    print("Total records:", len(records))
    print("Unique document groups:", total_groups)

    print("\nSplit distribution:")

    for name, split_records in [
        ("Training", train_records),
        ("Validation", val_records),
        ("Testing", test_records)
    ]:
        percentage = len(split_records) / len(records) * 100
        print(f"{name}: {len(split_records)} ({percentage:.1f}%)")

    print("\nOutput folder:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
