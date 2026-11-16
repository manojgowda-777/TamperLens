
import json
from pathlib import Path
from collections import Counter
from PIL import Image

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_ROOT = PROJECT_ROOT / "datasets" / "splits"

SPLIT_FILES = [
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl"
]


def main():

    dimensions = Counter()
    formats = Counter()

    total_images = 0
    failed_images = []

    min_width = float("inf")
    min_height = float("inf")

    max_width = 0
    max_height = 0

    for split_file in SPLIT_FILES:

        file_path = SPLIT_ROOT / split_file

        with open(file_path, "r", encoding="utf-8") as file:

            for line in file:

                if not line.strip():
                    continue

                record = json.loads(line)

                image_path = PROJECT_ROOT / record["image_path"]

                try:
                    with Image.open(image_path) as image:

                        width, height = image.size
                        image_format = image.format

                        dimensions[(width, height)] += 1
                        formats[image_format] += 1

                        total_images += 1

                        min_width = min(min_width, width)
                        min_height = min(min_height, height)

                        max_width = max(max_width, width)
                        max_height = max(max_height, height)

                except Exception as error:

                    failed_images.append(
                        (str(image_path), str(error))
                    )

    print("\nIMAGE INSPECTION REPORT")
    print("--------------------------------")

    print("Total images inspected:", total_images)
    print("Images that failed:", len(failed_images))

    print("\nIMAGE DIMENSIONS")
    print("--------------------------------")

    print("Minimum width:", min_width)
    print("Minimum height:", min_height)

    print("Maximum width:", max_width)
    print("Maximum height:", max_height)

    print("\nMOST COMMON DIMENSIONS")
    print("--------------------------------")

    for size, count in dimensions.most_common(10):
        print(f"{size[0]} x {size[1]} : {count} images")

    print("\nIMAGE FORMATS")
    print("--------------------------------")

    for image_format, count in formats.items():
        print(f"{image_format}: {count}")

    if failed_images:
        print("\nFAILED IMAGES")
        for path, error in failed_images[:10]:
            print(path, error)

    print("\nInspection completed!")


if __name__ == "__main__":
    main()
