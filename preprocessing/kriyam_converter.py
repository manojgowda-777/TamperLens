
import json
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

KRIYAM_ROOT = PROJECT_ROOT / "datasets" / "kriyam_tamperflow"

IMAGE_ROOT = KRIYAM_ROOT / "images"
ANNOTATION_ROOT = KRIYAM_ROOT / "annotations"
METADATA_FILE = KRIYAM_ROOT / "metadata.jsonl"

OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "standardized"
OUTPUT_FILE = OUTPUT_ROOT / "kriyam_metadata.jsonl"


def find_image(image_filename):
    """Find image using PNG, JPG, or JPEG extension."""

    original_path = KRIYAM_ROOT / image_filename

    if original_path.exists():
        return original_path

    image_stem = Path(image_filename).stem

    for extension in [".png", ".jpg", ".jpeg"]:
        candidate = IMAGE_ROOT / f"{image_stem}{extension}"

        if candidate.exists():
            return candidate

    return None


def load_kriyam_data():
    """Load Kriyam metadata and annotations."""

    records = []
    missing_annotations = 0
    missing_images = 0

    with open(METADATA_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            metadata = json.loads(line)

            document_id = metadata["id"]
            image_filename = metadata["file_name"]

            # Find image regardless of its extension
            image_path = find_image(image_filename)

            if image_path is None:
                missing_images += 1
                continue

            # Load document annotation
            annotation_file = ANNOTATION_ROOT / f"{document_id}.json"

            if not annotation_file.exists():
                missing_annotations += 1
                continue

            with open(annotation_file, "r", encoding="utf-8") as annotation_json:
                annotation = json.load(annotation_json)

            # Extract tampering regions
            tamper_regions = []

            for region in annotation.get("regions", []):

                tamper_regions.append({
                    "region_id": region.get("region_id"),
                    "x": region.get("x"),
                    "y": region.get("y"),
                    "width": region.get("w"),
                    "height": region.get("h"),
                    "tampering_types": region.get("tamper_types", []),
                    "ai_modified": region.get("ai_modified")
                })

            # Use document annotation for authenticity
            is_authentic = annotation.get("is_authentic")

            if is_authentic is None:
                is_authentic = metadata.get("is_authentic")

            authenticity = (
                "authentic" if is_authentic else "tampered"
            )

            # Extract tampering types
            tampering_types = metadata.get("tamper_types", [])

            if not tampering_types:
                for region in tamper_regions:
                    tampering_types.extend(
                        region.get("tampering_types", [])
                    )

            # Determine annotation level
            if tamper_regions:
                annotation_level = "tamper_bbox"
            else:
                annotation_level = "image_level"

            record = {
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "dataset": "Kriyam",
                "document_id": document_id,
                "authenticity": authenticity,
                "tampering_types": tampering_types,
                "source_image": None,
                "affected_fields": [],
                "field_boxes": [],
                "tamper_regions": tamper_regions,
                "annotation_level": annotation_level,

                # Preserve original Kriyam information
                "tier": metadata.get("tier"),
                "image_width": metadata.get("image_w"),
                "image_height": metadata.get("image_h"),
                "ai_generated_doc": metadata.get("ai_generated_doc"),
                "num_regions": metadata.get("num_regions")
            }

            records.append(record)

    return records, missing_annotations, missing_images


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    records, missing_annotations, missing_images = load_kriyam_data()

    # Save standardized JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in records:
            file.write(json.dumps(record) + "\n")

    # Statistics
    authentic_count = sum(
        1 for record in records
        if record["authenticity"] == "authentic"
    )

    tampered_count = sum(
        1 for record in records
        if record["authenticity"] == "tampered"
    )

    localized_count = sum(
        1 for record in records
        if record["tamper_regions"]
    )

    print("\nKriyam conversion completed!")
    print("--------------------------------")
    print("Total records:", len(records))
    print("Authentic records:", authentic_count)
    print("Tampered records:", tampered_count)
    print("Records with tamper coordinates:", localized_count)
    print("Missing annotations:", missing_annotations)
    print("Missing images:", missing_images)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
