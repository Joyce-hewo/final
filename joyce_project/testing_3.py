import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import torch
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (requires OPENAI_API_KEY in .env or env)
client = OpenAI()

# Configuration
JETSON_INFERENCE_ROOT = os.path.expanduser("~/jetson-inference")
ONNX_MODEL_PATH = os.path.join(JETSON_INFERENCE_ROOT, "python/training/classification/models/testdataset/resnet18.onnx")
LABELS_PATH = os.path.join(JETSON_INFERENCE_ROOT, "python/training/classification/data/Project_joyce/labels.txt")
IMAGE_PATH = os.path.join(JETSON_INFERENCE_ROOT, "python/training/classification/data/Project_joyce/test/CLOUDED_LEOPARD/3.jpg")
OUTPUT_IMAGE = "output.jpg"

# Machine 1: ONNX Image Classifier ======================================
def classify_image_onnx(image_path, model_path, labels_path):
    if not all([os.path.exists(p) for p in [model_path, labels_path, image_path]]):
        missing = [p for p in [model_path, labels_path, image_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing files: {missing}")

    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))

    img_array = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    input_tensor = np.expand_dims(img_array, axis=0)

    available_providers = ort.get_available_providers()
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options, providers=available_providers)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    outputs = sess.run([output_name], {input_name: input_tensor})

    probabilities = np.squeeze(outputs[0])
    top_class = np.argmax(probabilities)
    confidence = probabilities[top_class] * 100
    class_name = labels[top_class].upper()

    print(f"[image] loaded '{image_path}' ({img.size[0]}x{img.size[1]}, 3 channels)")
    print(f"imagenet: {confidence:.2f}% class #{top_class} ({class_name})")
    img_resized.save(OUTPUT_IMAGE)
    print(f"[image] saved '{OUTPUT_IMAGE}' (224x224, 3 channels)")

    return class_name

# Machine 2: OpenAI Recommendations =====================================
def generate_recommendations(english_word):
    try:
        prompt = f"""You are a helpful assistant that:
1. Translates English words to Chinese
2. Recommends related books/movies
3. Provides interesting facts
do not use any bold formatting

For the word: {english_word}
1. Chinese translation:
2. Recommended books/movies:
3. Interesting facts:
"""
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo" if you're on free tier
            messages=[
                {"role": "system", "content": "You are a multilingual assistant with deep knowledge of language and culture."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=0.9
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# Main Execution ========================================================
if __name__ == "__main__":
    try:
        print("\n=== Machine 1: ONNX Image Classification ===")
        classified_label = classify_image_onnx(IMAGE_PATH, ONNX_MODEL_PATH, LABELS_PATH)

        print("\n=== Machine 2: Translation & Recommendations ===")
        recommendations = generate_recommendations(classified_label)
        print("\nGenerated Output:\n" + recommendations)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
