
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

    total_images = 0

    orientation_counts = Counter()
    aspect_ratios = []

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

                        ratio = width / height

                        aspect_ratios.append(ratio)
                        total_images += 1

                        if ratio < 0.95:
                            orientation_counts["Portrait"] += 1

                        elif ratio > 1.05:
                            orientation_counts["Landscape"] += 1

                        else:
                            orientation_counts["Square"] += 1

                except Exception as error:
                    print("Error:", image_path, error)

    # Calculate statistics
    aspect_ratios.sort()

    average_ratio = sum(aspect_ratios) / len(aspect_ratios)

    median_ratio = aspect_ratios[len(aspect_ratios) // 2]

    print("\nIMAGE ASPECT RATIO REPORT")
    print("--------------------------------")

    print("Total images:", total_images)

    print("\nORIENTATION DISTRIBUTION")
    print("--------------------------------")

    for orientation, count in orientation_counts.items():

        percentage = count / total_images * 100

        print(
            f"{orientation}: {count} "
            f"({percentage:.2f}%)"
        )

    print("\nASPECT RATIO STATISTICS")
    print("--------------------------------")

    print(f"Average aspect ratio: {average_ratio:.3f}")
    print(f"Median aspect ratio: {median_ratio:.3f}")

    print(f"Minimum ratio: {min(aspect_ratios):.3f}")
    print(f"Maximum ratio: {max(aspect_ratios):.3f}")

    print("\nInspection completed!")


if __name__ == "__main__":
    main()
