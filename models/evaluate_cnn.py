
import torch
from torch.utils.data import DataLoader
from models.cnn_model import DocumentCNN
from preprocessing.torch_dataset import DocumentClassificationDataset

# Use CPU or GPU depending on availability
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load the test dataset
test_dataset = DocumentClassificationDataset("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

# Load our trained CNN
model = DocumentCNN().to(device)

model.load_state_dict(
    torch.load(
        "models/best_cnn_model.pth",
        map_location=device
    )
)

model.eval()

# Confusion matrix counters
true_authentic = 0
false_tampered = 0
false_authentic = 0
true_tampered = 0

# Evaluate the model
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        true_authentic += (
            ((predictions == 0) & (labels == 0)).sum().item()
        )

        false_tampered += (
            ((predictions == 1) & (labels == 0)).sum().item()
        )

        false_authentic += (
            ((predictions == 0) & (labels == 1)).sum().item()
        )

        true_tampered += (
            ((predictions == 1) & (labels == 1)).sum().item()
        )

# Calculate evaluation metrics
total = len(test_dataset)

correct = true_authentic + true_tampered

accuracy = (correct / total) * 100

precision = (
    true_tampered / (true_tampered + false_tampered)
    if (true_tampered + false_tampered) > 0 else 0
)

recall = (
    true_tampered / (true_tampered + false_authentic)
    if (true_tampered + false_authentic) > 0 else 0
)

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

print("\nCNN TEST RESULTS")
print("------------------------------")

print("Test images:", total)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Tampered Precision: {precision * 100:.2f}%")
print(f"Tampered Recall: {recall * 100:.2f}%")
print(f"Tampered F1 Score: {f1_score * 100:.2f}%")

print("\nConfusion Matrix")
print("------------------------------")
print("                 Predicted")
print("              Authentic  Tampered")
print(f"Actual Authentic  {true_authentic:6}  {false_tampered:6}")
print(f"Actual Tampered   {false_authentic:6}  {true_tampered:6}")

print("\nCNN evaluation completed!")
