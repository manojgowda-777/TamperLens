
import torch
import numpy as np

from pathlib import Path
from PIL import Image

from models.cnn_model import DocumentCNN
from models.unet_model import DocumentUNet

from preprocessing.image_preprocessor import preprocess_image


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

CNN_PATH = PROJECT_ROOT / "models" / "best_cnn_model.pth"
UNET_PATH = PROJECT_ROOT / "models" / "best_unet_bce_dice.pth"

UNET_THRESHOLD = 0.60

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. Load both trained models only once
# --------------------------------------------------

def load_models():

    print("Loading TamperLens models...")

    cnn = DocumentCNN().to(DEVICE)
    unet = DocumentUNet().to(DEVICE)

    cnn_checkpoint = torch.load(
        CNN_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    unet_checkpoint = torch.load(
        UNET_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    cnn.load_state_dict(cnn_checkpoint)
    unet.load_state_dict(unet_checkpoint)

    cnn.eval()
    unet.eval()

    print("CNN and U-Net loaded successfully.")

    return cnn, unet


# Load models once when the application starts
CNN_MODEL, UNET_MODEL = load_models()


# --------------------------------------------------
# 3. Prepare document image
# --------------------------------------------------

def prepare_image(image_path):

    # Apply the same preprocessing used during training
    processed_image = preprocess_image(image_path)

    # Normalize pixel values between 0 and 1
    image_array = (
        np.array(processed_image, dtype=np.float32) / 255.0
    )

    # Convert HWC format to CHW format
    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1).copy()
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE), np.array(processed_image)


# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

def predict_document(image_path):

    image_tensor, processed_image = prepare_image(image_path)

    with torch.no_grad():

        # CNN: Document authenticity classification
        cnn_output = CNN_MODEL(image_tensor)

        cnn_probabilities = torch.softmax(
            cnn_output,
            dim=1
        )

        authentic_probability = cnn_probabilities[0][0].item()
        tampered_probability = cnn_probabilities[0][1].item()

        cnn_prediction = torch.argmax(
            cnn_probabilities,
            dim=1
        ).item()

        # U-Net: Suspicious-region detection
        unet_output = UNET_MODEL(image_tensor)

        unet_probability = torch.sigmoid(unet_output)

        unet_mask = (
            unet_probability[0, 0] >= UNET_THRESHOLD
        ).cpu().numpy()

    # Calculate suspicious area percentage
    suspicious_area_percentage = (
        np.count_nonzero(unet_mask) / unet_mask.size
    ) * 100

    # Generate highlighted image
    highlighted_image = processed_image.copy()

    highlighted_image[unet_mask] = (
        highlighted_image[unet_mask] * 0.4
        + np.array([255, 0, 0]) * 0.6
    ).astype(np.uint8)

    # CNN independently decides the final classification
    if cnn_prediction == 0:

        final_result = "Likely Authentic"

    else:

        final_result = "Potentially Tampered"

    # U-Net independently reports region detection
    if suspicious_area_percentage > 0:

        region_status = "Suspicious regions highlighted"

    else:

        region_status = "No clear suspicious region identified"

    # Return all prediction results
    return {

        "cnn_prediction": (
            "Authentic" if cnn_prediction == 0 else "Tampered"
        ),

        "authentic_probability": authentic_probability,

        "tampered_probability": tampered_probability,

        "suspicious_area_percentage": suspicious_area_percentage,

        "final_result": final_result,

        "region_status": region_status,

        "highlighted_image": highlighted_image,

        "predicted_mask": unet_mask
    }


# --------------------------------------------------
# 5. Optional standalone testing
# --------------------------------------------------

if __name__ == "__main__":

    test_image = (
        PROJECT_ROOT
        / "datasets"
        / "processed"
        / "test"
        / "kriyam_kriyam_0023_348.png"
    )

    if not test_image.exists():

        print("Test image not found:", test_image)

    else:

        print("\nRunning TamperLens prediction...")

        results = predict_document(test_image)

        print("\nTAMPERLENS PREDICTION RESULTS")
        print("--------------------------------")

        print("CNN Prediction:", results["cnn_prediction"])

        print(
            "Authentic Probability:",
            round(results["authentic_probability"] * 100, 2),
            "%"
        )

        print(
            "Tampered Probability:",
            round(results["tampered_probability"] * 100, 2),
            "%"
        )

        print(
            "Suspicious Area:",
            round(results["suspicious_area_percentage"], 2),
            "%"
        )

        print("Final Result:", results["final_result"])

        print("Region Status:", results["region_status"])

        # Save highlighted image
        output_path = (
            PROJECT_ROOT / "models" / "prediction_highlighted.png"
        )

        Image.fromarray(
            results["highlighted_image"]
        ).save(output_path)

        print("\nHighlighted image saved at:", output_path)
