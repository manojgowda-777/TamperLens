
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed"


# Classification dataset
class DocumentClassificationDataset(Dataset):

    def __init__(self, split, augment=False):

        metadata_file = PROCESSED_ROOT / split / "metadata.jsonl"

        with open(metadata_file, "r", encoding="utf-8") as file:
            self.records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        # Enable augmentation only when requested
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=3),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10
            )
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = PROJECT_ROOT / record["processed_image_path"]

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            # Apply augmentation only to training images
            if self.augment:
                image = self.transform(image)

            image_array = (
                np.array(image, dtype=np.float32) / 255.0
            )

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        # 0 = Authentic, 1 = Tampered
        label = (
            1 if record["authenticity"] == "tampered" else 0
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return image_tensor, label_tensor


# Segmentation dataset
class TamperSegmentationDataset(Dataset):

    def __init__(self, split):

        metadata_file = (
            PROCESSED_ROOT / split / "segmentation_metadata.jsonl"
        )

        with open(metadata_file, "r", encoding="utf-8") as file:

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        # Keep only records with verified tampering-region masks
        self.records = [
            record for record in records
            if record["segmentation_eligible"]
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = PROJECT_ROOT / record["processed_image_path"]

        mask_path = PROJECT_ROOT / record["mask_path"]

        # Load document image without augmentation
        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image_array = (
                np.array(image, dtype=np.float32) / 255.0
            )

        # Load corresponding mask
        with Image.open(mask_path) as mask:

            mask = mask.convert("L")

            mask_array = (
                np.array(mask, dtype=np.float32) / 255.0
            )

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        # Convert mask values into 0 and 1
        mask_tensor = torch.from_numpy(
            (mask_array > 0.5).astype(np.float32)
        ).unsqueeze(0)

        return image_tensor, mask_tensor


# Test dataset loading
if __name__ == "__main__":

    print("TESTING PYTORCH DATASET")
    print("-------------------------------")

    # Test classification dataset with augmentation
    classification_data = DocumentClassificationDataset(
        "train",
        augment=True
    )

    segmentation_data = TamperSegmentationDataset("train")

    image, label = classification_data[0]

    print("\nClassification Dataset:")
    print("Total images:", len(classification_data))
    print("Image shape:", image.shape)
    print("Label:", label.item())

    image, mask = segmentation_data[0]

    print("\nSegmentation Dataset:")
    print("Eligible images:", len(segmentation_data))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mask values:", torch.unique(mask))

    print("\nPyTorch dataset loading successful!")
