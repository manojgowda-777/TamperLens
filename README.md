# TamperLens – AI-Powered Document Tampering Detection

TamperLens is an AI-powered document tampering detection system designed to identify potentially manipulated document images and highlight suspicious regions.

The system uses two custom-trained deep learning models: a **Convolutional Neural Network (CNN)** for document classification and a **U-Net** for suspicious-region localization.

## Project Overview

Digital document manipulation has become increasingly common, making it difficult to distinguish authentic documents from digitally modified ones.

TamperLens aims to address this problem by analyzing uploaded document images and providing:
- Document classification as Likely Authentic or Potentially Tampered.
- Authentic and tampered probability scores.
- Suspicious-region detection and visualization.
- An output image with detected regions highlighted in red.

The system is designed to support different document types, including certificates, marksheets, identity documents and other document images.

## Key Features

- **Document Upload:** Upload a document image for analysis.
- **AI-Based Classification:** Classifies documents using a custom-trained CNN.
- **Tampering Localization:** Uses a custom U-Net to identify potentially suspicious regions.
- **Visual Highlighting:** Displays detected regions with red overlays.
- **Prediction Results:** Shows classification probabilities and suspicious-area percentage.
- **Interactive Interface:** Provides a web-based interface for document analysis.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React.js, Vite, CSS |
| Backend | Python, Flask, Flask-CORS |
| Deep Learning | PyTorch |
| Models | Custom CNN, Custom U-Net |
| Image Processing | Pillow, NumPy |
| Development Tools | VS Code, Jupyter Notebook |

## Model Architecture

### 1. Custom CNN – Document Classification

The custom Convolutional Neural Network analyzes the uploaded document image and classifies it into two categories:

- Authentic
- Tampered

It also produces probability scores for both categories.

### 2. Custom U-Net – Suspicious Region Localization

The custom U-Net performs image segmentation to identify potentially suspicious regions within the document.

The predicted regions are converted into a mask and overlaid in red on the original document image.

**Important:** The CNN performs classification, while the U-Net independently performs suspicious-region localization.

## System Workflow

1. User uploads a document image through the frontend.
2. The React frontend sends the image to the Flask backend.
3. The image is preprocessed into the required format.
4. The CNN predicts whether the document is likely authentic or potentially tampered.
5. The U-Net independently predicts suspicious regions.
6. The backend generates a highlighted output image.
7. The frontend displays the classification results and highlighted image.

## Project Structure

```text
TamperLens/
│
├── app.py
├── prediction.py
├── .gitignore
│
├── models/
│   ├── cnn_model.py
│   ├── unet_model.py
│   ├── best_cnn_model.pth
│   ├── best_unet_bce_dice.pth
│   ├── train_cnn.py
│   ├── train_unet_bce_dice.py
│   ├── evaluate_cnn.py
│   └── evaluate_unet_final.py
│
├── preprocessing/
│   ├── dataset_combiner.py
│   ├── image_preprocessor.py
│   ├── mask_generator.py
│   ├── split_dataset.py
│   ├── torch_dataset.py
│   └── other preprocessing scripts
│
└── frontend/
    ├── public/
    ├── src/
    ├── package.json
    └── vite.config.js
```

## Installation and Setup

### Prerequisites

- Python
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/manojgowda-777/TamperLens.git
cd TamperLens
```

### 2. Backend Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required Python packages:

```bash
pip install flask flask-cors torch numpy pillow
```

Start the Flask backend:

```bash
python app.py
```

Backend URL:

```text
http://127.0.0.1:5000
```

### 3. Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```



