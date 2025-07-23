from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from testing_3 import classify_image_onnx, generate_recommendations  # Update import as needed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['ALLOWED_EXTENSIONS'] = {"jpg", "jpeg", "png"}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

JETSON_INFERENCE_ROOT = os.path.expanduser("~/jetson-inference")
ONNX_MODEL_PATH = os.path.join(JETSON_INFERENCE_ROOT, "python/training/classification/models/testdataset/resnet18.onnx")
LABELS_PATH = os.path.join(JETSON_INFERENCE_ROOT, "python/training/classification/data/Project_joyce/labels.txt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Run classification and get recommendations
            classified_label = classify_image_onnx(image_path, ONNX_MODEL_PATH, LABELS_PATH)
            recommendations = generate_recommendations(classified_label)

            return render_template("index.html", label=classified_label, recommendations=recommendations)

        return render_template("index.html", label=None, recommendations="Invalid file type. Please upload a .jpg or .png.")

    return render_template("index.html", label=None, recommendations=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
