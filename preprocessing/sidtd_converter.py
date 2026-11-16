
import json
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIDTD_ROOT = PROJECT_ROOT / "datasets" / "SIDTD" / "templates"

IMAGE_ROOT = SIDTD_ROOT / "Images"
ANNOTATION_ROOT = SIDTD_ROOT / "Annotations"

OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "standardized"
OUTPUT_FILE = OUTPUT_ROOT / "sidtd_metadata.jsonl"


def find_real_image(filename):
    """Find the actual genuine image using its filename."""

    real_image_dir = IMAGE_ROOT / "reals"
    original_name = Path(filename).name

    # First, check the exact filename
    exact_path = real_image_dir / original_name

    if exact_path.is_file():
        return exact_path

    # If not found, search using the filename suffix
    matches = list(real_image_dir.glob(f"*{original_name}"))

    if len(matches) == 1:
        return matches[0]

    return None


def load_real_annotations():
    """Load genuine SIDTD VIA annotations."""

    records = []
    missing_images = 0

    real_annotation_dir = ANNOTATION_ROOT / "reals"

    for json_file in real_annotation_dir.glob("*.json"):

        # Get category from annotation filename
        category = json_file.stem

        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        image_metadata = data.get("_via_img_metadata", {})

        for image_data in image_metadata.values():

            filename = image_data.get("filename")

            if not filename:
                continue

            # Add category prefix to numbered filename
            image_filename = f"{category}_{Path(filename).stem}.jpg"

            image_path = IMAGE_ROOT / "reals" / image_filename

            if not image_path.is_file():
                missing_images += 1
                print(f"Missing genuine image: {image_filename}")
                continue

            field_boxes = []

            for region in image_data.get("regions", []):

                shape = region.get("shape_attributes", {})
                attributes = region.get("region_attributes", {})

                if shape.get("name") != "rect":
                    continue

                field_boxes.append({
                    "x": shape.get("x"),
                    "y": shape.get("y"),
                    "width": shape.get("width"),
                    "height": shape.get("height"),
                    "field_name": attributes.get("field_name"),
                    "value": attributes.get("value")
                })

            records.append({
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "dataset": "SIDTD",
                "document_id": image_path.stem,
                "authenticity": "authentic",
                "tampering_types": [],
                "source_image": None,
                "affected_fields": [],
                "field_boxes": field_boxes,
                "tamper_regions": [],
                "annotation_level": "field_bbox"
            })

    print("Missing genuine images:", missing_images)

    return records

def load_fake_annotations():
    """Load fake SIDTD annotation files."""

    records = []

    fake_annotation_dir = ANNOTATION_ROOT / "fakes"

    for json_file in fake_annotation_dir.glob("*.json"):

        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        annotation_name = data.get("name")

        if not annotation_name:
            continue

        image_filename = annotation_name + ".jpg"
        image_path = IMAGE_ROOT / "fakes" / image_filename

        if not image_path.exists():
            print(f"Missing fake image: {image_filename}")
            continue

        affected_field = data.get("field")

        records.append({
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "dataset": "SIDTD",
            "document_id": annotation_name,
            "authenticity": "tampered",
            "tampering_types": [data.get("ctype")],
            "source_image": data.get("src"),
            "affected_fields": (
                [affected_field]
                if affected_field and affected_field != "None"
                else []
            ),
            "field_boxes": [],
            "tamper_regions": [],
            "annotation_level": "tampering_metadata"
        })

    return records


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    real_records = load_real_annotations()
    fake_records = load_fake_annotations()

    all_records = real_records + fake_records

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in all_records:
            file.write(json.dumps(record) + "\n")

    print("\nSIDTD conversion completed!")
    print("--------------------------------")
    print("Authentic records:", len(real_records))
    print("Tampered records:", len(fake_records))
    print("Total records:", len(all_records))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
