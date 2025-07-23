# Classilate

<img width="300" height="300" alt="ChatGPT Image 2025年7月23日 11_09_51" src="https://github.com/user-attachments/assets/a6dde061-6e3c-4976-ab1f-0375d5053647" />

A Flask web app for image classification and AI-powered enrichment using Jetson and OpenAI's language model. Classilate takes an image, classifies its content using an ONNX image model, then translates the result to Chinese and recommends relevant media using the OpenAI API.

## The Algorithm

This project uses a two-part AI system to analyze images and generate meaningful linguistic content based on what’s identified in the image:

- **Machine 1**: An image classification system powered by Jetson Inference and a ResNet18 ONNX model. It identifies the object in the image and outputs a class label with a confidence score. Example:
imagenet.py
--model=$NET/resnet18.onnx
--labels=$DATASET/labels.txt
--input_blob=input_0
--output_blob=output_0
$DATASET/test/CLOUDED_LEOPARD/3.jpg output.jpg

Output:
[image] loaded '.../CLOUDED_LEOPARD/3.jpg' (224x224, 3 channels)
imagenet: 78.51% class #4 (CLOUDED LEOPARD)
[image] saved 'output.jpg' (224x224, 3 channels)

- **Machine 2**: A language processing module that sends the identified class (e.g., “Clouded Leopard”) to a prompt using the OpenAI API. The prompt format is:
"You are a helpful assistant that:
1. Translates English words to Chinese
2. Recommends related books/movies
3. Provides interesting facts
For the word: CLOUDED LEOPARD
1. Chinese translation:
2. Recommended books/movies:
3. Interesting facts:

(Originally tested using LLaMA (via HuggingFace and Ollama), the final implementation uses OpenAI's GPT models via API key for more accurate and stable responses.)

## Running this Project

1. Download or clone this repository and place it on your Jetson or development machine.
2. Ensure your image classification script (`imagenet.py`) and ONNX model are functional.
3. Set up your environment:
 ```bash
 pip3 install flask openai numpy==1.24.4 ```bash
4. Export your OpenAI API key:
export OPENAI_API_KEY=your-api-key-here
5. Run the app:
python3 app.py
6. Open the link to the Web UI in your browser (usually appears in your terminal or bottom corner).
7. Upload a sample image via the “Choose File” button.
8. Click the Classify Image button to start the two-part process:
9. Machine 1 classifies the image
10. Machine 2 generates translation + recommendations + facts
11. The result will be displayed on screen:
- The image
- Classification label and confidence
- Chinese translation
- Book/movie suggestions
 -Interesting facts
