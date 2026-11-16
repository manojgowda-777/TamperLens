
from pathlib import Path
from uuid import uuid4

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from PIL import Image, UnidentifiedImageError

from prediction import predict_document


# --------------------------------------------------
# 1. Flask configuration
# --------------------------------------------------

app = Flask(__name__)

CORS(app)

PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
RESULT_FOLDER = PROJECT_ROOT / "results"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MAX_FILE_SIZE = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# --------------------------------------------------
# 2. Check allowed file extensions
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# 3. Home API
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "message": "Welcome to TamperLens API",

        "status": "Backend is running"

    })


# --------------------------------------------------
# 4. Document upload and prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:

        return jsonify({
            "error": "No document image uploaded"
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":

        return jsonify({
            "error": "Please select an image"
        }), 400

    if not allowed_file(uploaded_file.filename):

        return jsonify({
            "error": "Only PNG, JPG and JPEG images are allowed"
        }), 400

    unique_id = uuid4().hex

    upload_path = UPLOAD_FOLDER / f"{unique_id}.png"

    result_filename = f"{unique_id}_highlighted.png"

    result_path = RESULT_FOLDER / result_filename

    try:

        # Validate uploaded image
        with Image.open(uploaded_file.stream) as image:

            image.verify()

        # Reset file stream before reopening
        uploaded_file.stream.seek(0)

        # Convert uploaded image to RGB PNG
        with Image.open(uploaded_file.stream) as image:

            image = image.convert("RGB")

            image.save(upload_path, format="PNG")

        # Run CNN and U-Net prediction
        results = predict_document(upload_path)

        # Save highlighted image
        Image.fromarray(
            results["highlighted_image"]
        ).save(result_path, format="PNG")

        # Return prediction information
        return jsonify({

            "success": True,

            "cnn_prediction": results["cnn_prediction"],

            "authentic_probability": round(
                results["authentic_probability"] * 100, 2
            ),

            "tampered_probability": round(
                results["tampered_probability"] * 100, 2
            ),

            "suspicious_area_percentage": round(
                results["suspicious_area_percentage"], 2
            ),

            "final_result": results["final_result"],

            "region_status": results["region_status"],

            "highlighted_image_url": f"/results/{result_filename}"

        }), 200

    except (UnidentifiedImageError, OSError, ValueError):

        return jsonify({

            "error": "Invalid or unsupported image file"

        }), 400

    except Exception:

        app.logger.exception("Document prediction failed")

        return jsonify({

            "error": "Prediction failed. Please try another image."

        }), 500

    finally:

        # Delete temporary uploaded image
        if upload_path.exists():

            upload_path.unlink()


# --------------------------------------------------
# 5. Display highlighted image
# --------------------------------------------------

@app.route("/results/<filename>", methods=["GET"])
def get_result(filename):

    if (
        not filename.endswith("_highlighted.png")
        or Path(filename).name != filename
    ):

        return jsonify({
            "error": "Invalid result filename"
        }), 400

    return send_from_directory(
        RESULT_FOLDER,
        filename,
        mimetype="image/png"
    )


# --------------------------------------------------
# 6. Run Flask server
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting TamperLens Flask backend...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
