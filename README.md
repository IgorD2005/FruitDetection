# Fruit Image Classifier - CNN

A convolutional neural network trained to classify three fruit categories: **banana**, **orange**, and **lemon**. Built with **TensorFlow/Keras** as part of a computer vision course.

---

## Model Architecture

The network consists of **four convolutional blocks**, each containing:

- `Conv2D`
- `MaxPooling2D`

The number of filters increases progressively:

- 16
- 32
- 64
- 64

This architecture enables the model to learn increasingly complex visual features.

After the convolutional layers:

- `Dense(128)`
- `Dropout(0.3)`
- `Softmax` output layer (3 classes)

**Input shape:** `128 × 128 × 3`

### Training Configuration

- **Optimizer:** Adam (`learning_rate = 0.0001`)
- **Loss Function:** Categorical Crossentropy
- **Regularization:**
  - Dropout (`0.3`)
  - EarlyStopping (`patience = 5`)

---

## Dataset

The dataset contains approximately **480 images per class** across three categories:

- 🍌 Banana
- 🍊 Orange
- 🍋 Lemon

Dataset split:

- **80%** Training
- **20%** Validation

Data augmentation was applied **only to the training set**, including:

- Rotation
- Zoom
- Horizontal Flip
- Brightness Variation
- Shear

---

## Results

The model achieved approximately **85% validation accuracy** after training for up to **20 epochs** with **Early Stopping** enabled.

The best-performing model was automatically saved using **ModelCheckpoint**.

---

## Project Structure

```text
.
├── lab5.py                   # Training and inference pipeline
├── dataset/                  # Images organized by class
├── best_model.keras          # Saved model weights
├── historia_treningu.png     # Training & validation accuracy/loss
└── wynik_klasyfikacji.png    # Example classification result
```

---

## Requirements

- TensorFlow
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## Usage

### Train the model

```bash
python lab5.py
```

### Classify a single image

1. Place the image in the project root.
2. Rename it to:

```text
test.jpg
```

3. Run:

```bash
python lab5.py
```

The script will load the saved model and display the predicted fruit class.
