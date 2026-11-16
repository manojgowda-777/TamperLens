
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "standardized"
    / "combined_metadata.jsonl"
)

valid_records = 0
invalid_records = 0
missing_images = 0
invalid_labels = 0
missing_document_ids = 0
invalid_boxes = 0

authentic_count = 0
tampered_count = 0

with open(DATASET_FILE, "r", encoding="utf-8") as file:

    for line_number, line in enumerate(file, start=1):

        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            invalid_records += 1
            continue

        valid_records += 1

        # Check image path
        image_path = PROJECT_ROOT / record.get("image_path", "")

        if not image_path.is_file():
            missing_images += 1

        # Check authenticity label
        label = record.get("authenticity")

        if label == "authentic":
            authentic_count += 1
        elif label == "tampered":
            tampered_count += 1
        else:
            invalid_labels += 1

        # Check document ID
        if not record.get("document_id"):
            missing_document_ids += 1

        # Check bounding boxes
        for box in record.get("tamper_regions", []):

            x = box.get("x")
            y = box.get("y")
            width = box.get("width")
            height = box.get("height")

            if (
                not all(
                    isinstance(value, (int, float))
                    for value in [x, y, width, height]
                )
                or width <= 0
                or height <= 0
            ):
                invalid_boxes += 1


print("\nDATASET VALIDATION REPORT")
print("--------------------------------")
print("Valid JSON records:", valid_records)
print("Invalid JSON records:", invalid_records)
print("Missing images:", missing_images)
print("Invalid labels:", invalid_labels)
print("Missing document IDs:", missing_document_ids)
print("Invalid bounding boxes:", invalid_boxes)

print("\nCLASS DISTRIBUTION")
print("--------------------------------")
print("Authentic:", authentic_count)
print("Tampered:", tampered_count)

print("\nValidation completed!")
