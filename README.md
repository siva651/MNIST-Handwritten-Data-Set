# MNIST-Handwritten-Data-Set
# Handwritten Digit Classifier (MNIST CNN)

A convolutional neural network that classifies handwritten digits (0–9), trained on MNIST and wrapped in an interactive Gradio demo where I can draw a digit or upload an image and get a live prediction. I also added a multi-digit mode so I can recognize short sequences like phone numbers.

## What this project does

1. Loads and explores the MNIST dataset (60,000 training images, 10,000 test images)
2. Trains a CNN with batch normalization and dropout for regularization
3. Evaluates the model in depth — confusion matrix, per-class accuracy, misclassified examples
4. Visualizes what the first convolutional layer actually "sees" via feature maps
5. Saves the trained model to disk
6. Serves an interactive Gradio app for drawing/uploading a single digit
7. Extends that into a second app that segments and recognizes multiple digits in one image

## Model architecture

A `Sequential` CNN (`MNIST_CNN`) with three convolutional blocks of increasing depth:

- Block 1: two Conv2D(32) layers → BatchNorm → MaxPool → Dropout(0.25)
- Block 2: two Conv2D(64) layers → BatchNorm → MaxPool → Dropout(0.25)
- Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
- Dense(256) → BatchNorm → Dropout(0.5) → Dense(10, softmax)

Trained with the Adam optimizer and sparse categorical cross-entropy, using early stopping and learning-rate reduction on plateau so it doesn't overfit or waste epochs.

## Results

The model reaches high accuracy (typically 99%+) on the MNIST test set. The notebook prints the final test accuracy and generates:

- A confusion matrix heatmap
- A full `classification_report` (precision/recall/F1 per digit)
- A per-class accuracy bar chart
- A grid of the model's actual misclassified examples, which is honestly the most useful debugging tool in the whole notebook

## Interpretability

I pull the activations from the first Conv2D layer for a sample test image and plot the resulting feature maps, so I can see what low-level patterns (edges, strokes, curves) the network is picking up on before it gets to any higher-level reasoning.

## Interactive demo

The Gradio app has two ways to get a digit in:

- **Draw tab** — a sketchpad where I draw a digit by hand
- **Upload tab** — upload or paste an image of a digit

Both feed into the same preprocessing pipeline before hitting the model:

- Flatten any transparency onto white (fixes a common bug where transparent pixels turn black)
- Convert to grayscale
- Auto-invert if needed (MNIST expects a light digit on a dark background)
- Crop to the digit's bounding box, resize, and re-center using its center of mass — this mimics how MNIST digits were originally prepared, and it matters a lot for accuracy on hand-drawn input

The output shows a bar chart of the model's confidence across all 10 digits, plus a preview of exactly what the model sees after preprocessing (the 28x28 input), which makes it a lot easier to tell if a bad prediction is a preprocessing issue or a genuine model mistake.

## Multi-digit recognition

For sequences (like a phone number), I use OpenCV to:

1. Threshold and dilate the image so broken strokes merge into single blobs
2. Find contours and their bounding boxes
3. Sort the boxes left to right
4. Crop and run each one through the same single-digit preprocessing + prediction pipeline
5. Draw labeled bounding boxes over the original image and stitch the predicted digits into one string

## Setup

```bash
pip install tensorflow gradio scikit-learn seaborn matplotlib opencv-python scipy pillow
```

Run the notebook/script top to bottom. It will:

- Train the model (or load `mnist_cnn.keras` from disk if one already exists — see Step 8's sanity check)
- Launch the single-digit Gradio app
- Launch the multi-digit Gradio app

Both apps launch with `share=True`, so a public link is generated in addition to the local one.

## Files

- `mnist_cnn.keras` — the saved trained model

