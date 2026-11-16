
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_ROOT = PROJECT_ROOT / "datasets" / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "processed"

# Initial model input size
TARGET_SIZE = 512

SPLIT_FILES = [
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl"
]


def preprocess_image(image_path):

    with Image.open(image_path) as image:

        # Convert image to RGB
        image = image.convert("RGB")

        # Preserve aspect ratio while resizing
        image.thumbnail(
            (TARGET_SIZE, TARGET_SIZE),
            Image.Resampling.LANCZOS
        )

        # Create square canvas
        canvas = Image.new(
            "RGB",
            (TARGET_SIZE, TARGET_SIZE),
            (255, 255, 255)
        )

        # Center the resized image
        x = (TARGET_SIZE - image.width) // 2
        y = (TARGET_SIZE - image.height) // 2

        canvas.paste(image, (x, y))

        return canvas


def main():

    total_processed = 0
    failed_images = []

    for split_file in SPLIT_FILES:

        split_name = Path(split_file).stem

        input_file = SPLIT_ROOT / split_file
        output_folder = OUTPUT_ROOT / split_name

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(input_file, "r", encoding="utf-8") as file:

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        output_metadata = []

        for index, record in enumerate(records):

            image_path = PROJECT_ROOT / record["image_path"]

            try:

                processed_image = preprocess_image(image_path)

                # Use a unique filename for each record
                output_name = (
                    f"{record['dataset'].lower()}_"
                    f"{record['document_id']}_"
                    f"{index}.png"
                )

                output_path = output_folder / output_name

                processed_image.save(
                    output_path,
                    format="PNG"
                )

                # Keep original metadata and add processed path
                updated_record = record.copy()

                updated_record["processed_image_path"] = str(
                    output_path.relative_to(PROJECT_ROOT)
                )

                output_metadata.append(updated_record)

                total_processed += 1

            except Exception as error:

                failed_images.append(
                    (str(image_path), str(error))
                )

        # Save metadata for each split
        metadata_path = (
            output_folder / "metadata.jsonl"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            for record in output_metadata:

                file.write(
                    json.dumps(record) + "\n"
                )

        print(
            f"{split_name}: "
            f"{len(output_metadata)} images processed"
        )

    print("\nIMAGE PREPROCESSING REPORT")
    print("--------------------------------")

    print("Total processed:", total_processed)
    print("Failed images:", len(failed_images))

    if failed_images:

        print("\nFailed image details:")

        for path, error in failed_images[:10]:
            print(path, error)

    print("\nOutput folder:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
