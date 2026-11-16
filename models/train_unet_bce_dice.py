
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from models.unet_model import DocumentUNet
from preprocessing.torch_dataset import TamperSegmentationDataset


# 1. Use CPU (same setup as original training)
device = torch.device("cpu")

print("Starting experimental U-Net training...")
print("Device:", device)


# 2. Load existing datasets
train_dataset = TamperSegmentationDataset("train")
val_dataset = TamperSegmentationDataset("validation")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# 3. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# 4. Create a fresh U-Net model
model = DocumentUNet().to(device)


# 5. Define combined BCE + Dice Loss
bce_criterion = nn.BCEWithLogitsLoss()


def dice_loss(outputs, masks):
    # Convert predictions into probabilities
    probabilities = torch.sigmoid(outputs)

    # Calculate overlap between predictions and actual masks
    smooth = 1e-6

    intersection = (probabilities * masks).sum(dim=(1, 2, 3))

    total = (
        probabilities.sum(dim=(1, 2, 3))
        + masks.sum(dim=(1, 2, 3))
    )

    dice_score = (2 * intersection + smooth) / (total + smooth)

    return 1 - dice_score.mean()


def combined_loss(outputs, masks):
    bce = bce_criterion(outputs, masks)
    dice = dice_loss(outputs, masks)

    return bce + dice


# 6. Optimizer (same as original)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 7. Training settings
epochs = 10
best_val_loss = float("inf")

# IMPORTANT: Save to a new file
save_path = Path("models/best_unet_bce_dice.pth")


# 8. Start training
for epoch in range(epochs):

    model.train()
    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = combined_loss(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    average_train_loss = train_loss / len(train_loader)


    # 9. Validate the model
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = combined_loss(outputs, masks)

            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Training Loss: {average_train_loss:.4f} "
        f"Validation Loss: {average_val_loss:.4f}"
    )


    # 10. Save the best experimental model
    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        torch.save(model.state_dict(), save_path)

        print("Best experimental U-Net saved!")


print("\nExperimental training completed!")
print("Best validation loss:", round(best_val_loss, 4))
print("Model saved at:", save_path)
