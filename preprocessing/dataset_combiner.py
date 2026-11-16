
import json
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STANDARDIZED_ROOT = PROJECT_ROOT / "datasets" / "standardized"

SIDTD_FILE = STANDARDIZED_ROOT / "sidtd_metadata.jsonl"
KRIYAM_FILE = STANDARDIZED_ROOT / "kriyam_metadata.jsonl"

OUTPUT_FILE = STANDARDIZED_ROOT / "combined_metadata.jsonl"


def load_jsonl(file_path):
    """Load records from a JSONL file."""

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    # Load both datasets
    sidtd_records = load_jsonl(SIDTD_FILE)
    kriyam_records = load_jsonl(KRIYAM_FILE)

    # Combine records
    combined_records = sidtd_records + kriyam_records

    # Save combined dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in combined_records:
            file.write(json.dumps(record) + "\n")

    # Display statistics
    authentic_count = sum(
        1 for r in combined_records
        if r["authenticity"] == "authentic"
    )

    tampered_count = sum(
        1 for r in combined_records
        if r["authenticity"] == "tampered"
    )

    print("\nDataset combination completed!")
    print("--------------------------------")
    print("SIDTD records:", len(sidtd_records))
    print("Kriyam records:", len(kriyam_records))
    print("Total records:", len(combined_records))
    print("Authentic records:", authentic_count)
    print("Tampered records:", tampered_count)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
