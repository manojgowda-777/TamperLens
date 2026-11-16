
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = (
    PROJECT_ROOT / "datasets" / "processed" /
    "train" / "segmentation_metadata.jsonl"
)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file]

eligible_records = [
    record for record in records
    if record["segmentation_eligible"]
]

samples = random.sample(eligible_records, 3)

for record in samples:
    image_path = PROJECT_ROOT / record["processed_image_path"]
    mask_path = PROJECT_ROOT / record["mask_path"]

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    overlay = image.copy()
    overlay[mask > 0] = (
        0.5 * overlay[mask > 0] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Document")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Generated Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
