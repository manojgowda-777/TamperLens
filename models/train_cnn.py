
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.cnn_model import DocumentCNN
from preprocessing.torch_dataset import DocumentClassificationDataset

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training settings
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

# Load datasets
train_dataset = DocumentClassificationDataset(
    "train", augment=True
)

val_dataset = DocumentClassificationDataset(
    "validation", augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Create model
model = DocumentCNN().to(device)

# Calculate class weights to handle class imbalance
labels = [
    record["authenticity"]
    for record in train_dataset.records
]

authentic_count = labels.count("authentic")
tampered_count = labels.count("tampered")

total = len(labels)

class_weights = torch.tensor(
    [
        total / (2 * authentic_count),
        total / (2 * tampered_count)
    ],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = 0.0

print("Starting CNN training...")
print("Device:", device)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# Training loop
for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # Validation
    model.eval()

    correct = 0
    total_val = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total_val += labels.size(0)

    accuracy = 100 * correct / total_val

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

    # Save the best model
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "models/best_cnn_model.pth"
        )

print("\nTraining completed!")
print("Best validation accuracy:", round(best_accuracy, 2), "%")
print("Best model saved successfully!")
