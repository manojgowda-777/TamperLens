
import torch
from torch.utils.data import DataLoader

from models.unet_model import DocumentUNet
from preprocessing.torch_dataset import TamperSegmentationDataset


# 1. Select CPU
device = torch.device("cpu")

# 2. Set the selected validation threshold
THRESHOLD = 0.60

# 3. Load the test dataset
test_dataset = TamperSegmentationDataset("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# 4. Load experimental U-Net
model = DocumentUNet().to(device)

checkpoint = torch.load(
    "models/best_unet_bce_dice.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

print("Final U-Net evaluation started...")
print("Device:", device)
print("Threshold:", THRESHOLD)
print("Test images:", len(test_dataset))


# 5. Initialize counters
TP = 0
FP = 0
FN = 0
TN = 0


# 6. Evaluate test images
with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        TP += ((predictions == 1) & (masks == 1)).sum().item()
        FP += ((predictions == 1) & (masks == 0)).sum().item()
        FN += ((predictions == 0) & (masks == 1)).sum().item()
        TN += ((predictions == 0) & (masks == 0)).sum().item()


# 7. Calculate metrics
epsilon = 1e-8

accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)

precision = TP / (TP + FP + epsilon)

recall = TP / (TP + FN + epsilon)

dice = (2 * TP) / (2 * TP + FP + FN + epsilon)

iou = TP / (TP + FP + FN + epsilon)


# 8. Display final results
print("\n========== FINAL U-NET TEST RESULTS ==========")

print(f"Threshold      : {THRESHOLD}")
print(f"Pixel Accuracy : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"Dice Score     : {dice * 100:.2f}%")
print(f"IoU Score      : {iou * 100:.2f}%")

print("\nConfusion Counts:")
print("True Positive :", TP)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Negative :", TN)

print("\nFinal U-Net evaluation completed!")
