
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed"

TARGET_SIZE = 512

SPLITS = ["train", "validation", "test"]


def create_mask(record):

    original_path = PROJECT_ROOT / record["image_path"]

    # Get original image dimensions and reproduce preprocessing geometry
    with Image.open(original_path) as original:
        original = original.convert("RGB")

        original_width, original_height = original.size

        original.thumbnail(
            (TARGET_SIZE, TARGET_SIZE),
            Image.Resampling.LANCZOS
        )

        resized_width, resized_height = original.size

    # Same centered padding used by image_preprocessor.py
    offset_x = (TARGET_SIZE - resized_width) // 2
    offset_y = (TARGET_SIZE - resized_height) // 2

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    mask = np.zeros(
        (TARGET_SIZE, TARGET_SIZE),
        dtype=np.uint8
    )

    regions = record.get("tamper_regions", [])

    for region in regions:

        x = float(region["x"])
        y = float(region["y"])
        width = float(region["width"])
        height = float(region["height"])

        if width <= 0 or height <= 0:
            raise ValueError("Invalid bounding box dimensions")

        # Clip box to original image boundaries
        left = max(0, x)
        top = max(0, y)

        right = min(original_width, x + width)
        bottom = min(original_height, y + height)

        if right <= left or bottom <= top:
            raise ValueError("Bounding box outside image")

        # Transform original coordinates to processed image coordinates
        x1 = offset_x + math.floor(left * scale_x)
        y1 = offset_y + math.floor(top * scale_y)

        x2 = offset_x + math.ceil(right * scale_x)
        y2 = offset_y + math.ceil(bottom * scale_y)

        # Clip to 512 x 512 canvas
        x1 = max(0, min(TARGET_SIZE, x1))
        y1 = max(0, min(TARGET_SIZE, y1))
        x2 = max(0, min(TARGET_SIZE, x2))
        y2 = max(0, min(TARGET_SIZE, y2))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    return Image.fromarray(mask, mode="L")


def main():

    total_masks = 0
    skipped_records = 0
    failed_records = 0

    for split in SPLITS:

        split_folder = PROCESSED_ROOT / split

        input_file = split_folder / "metadata.jsonl"

        mask_folder = split_folder / "masks"
        mask_folder.mkdir(parents=True, exist_ok=True)

        output_metadata = []

        with open(input_file, "r", encoding="utf-8") as file:

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        for record in records:

            updated_record = record.copy()

            updated_record["segmentation_eligible"] = False
            updated_record["mask_path"] = None

            # Only use records with actual tampering-region annotations
            eligible = (
                str(record.get("dataset", "")).lower() == "kriyam"
                and record.get("authenticity") == "tampered"
                and record.get("annotation_level") == "tamper_bbox"
                and bool(record.get("tamper_regions"))
            )

            if not eligible:
                skipped_records += 1
                output_metadata.append(updated_record)
                continue

            try:

                mask = create_mask(record)

                processed_path = Path(
                    record["processed_image_path"]
                )

                mask_name = processed_path.stem + "_mask.png"

                mask_path = mask_folder / mask_name

                mask.save(mask_path, format="PNG")

                updated_record["segmentation_eligible"] = True

                updated_record["mask_path"] = str(
                    mask_path.relative_to(PROJECT_ROOT)
                )

                total_masks += 1

            except Exception as error:

                failed_records += 1

                print(
                    "Mask generation failed:",
                    record.get("document_id"),
                    record.get("tier"),
                    error
                )

            output_metadata.append(updated_record)

        output_file = (
            split_folder / "segmentation_metadata.jsonl"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            for record in output_metadata:
                file.write(json.dumps(record) + "\n")

        print(f"\n{split.upper()} SPLIT")
        print("--------------------------------")
        print("Total records:", len(records))
        print(
            "Eligible masks:",
            sum(r["segmentation_eligible"] for r in output_metadata)
        )

    print("\nMASK GENERATION REPORT")
    print("--------------------------------")
    print("Total masks generated:", total_masks)
    print("Records without eligible masks:", skipped_records)
    print("Failed mask generations:", failed_records)

    print("\nOutput folder:", PROCESSED_ROOT)


if __name__ == "__main__":
    main()
