
import torch
import torch.nn as nn


class DocumentCNN(nn.Module):

    def __init__(self):

        super(DocumentCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = DocumentCNN()

    sample = torch.randn(1, 3, 512, 512)

    output = model(sample)

    print("CNN model created successfully!")
    print("Output shape:", output.shape)

    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters())
    )
