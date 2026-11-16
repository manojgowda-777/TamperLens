
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_ROOT = (
    PROJECT_ROOT / "datasets" / "kriyam_tamperflow" / "annotations"
)

# Inspect one annotation for each of these types
target_ids = [
    "kriyam_0205",
    "kriyam_0624",
    "kriyam_0635"
]

for document_id in target_ids:

    annotation_file = ANNOTATION_ROOT / f"{document_id}.json"

    print("\n" + "=" * 55)
    print("DOCUMENT:", document_id)
    print("=" * 55)

    if not annotation_file.exists():
        print("Annotation file not found:", annotation_file)
        continue

    with open(annotation_file, "r", encoding="utf-8") as file:
        annotation = json.load(file)

    print("\nTop-level keys:")
    print(list(annotation.keys()))
    print("\nNotes:")
    print(json.dumps(annotation.get("notes"), indent=4))
    regions = annotation.get("regions", [])

    print("\nNumber of regions:", len(regions))

    for index, region in enumerate(regions[:5]):

        print(f"\nRegion {index + 1}:")
        print(json.dumps(region, indent=4))

print("\nAnnotation inspection completed!")
