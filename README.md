Fruit Image Classifier - CNN

A convolutional neural network trained to classify three fruit categories: banana, orange, and lemon. Built with TensorFlow/Keras as part of a computer vision course at the Polish-Japanese Academy of Information Technology.


Model Architecture

Four convolutional blocks are stacked sequentially, each consisting of a Conv2D layer followed by MaxPooling2D. Filter counts increase from 16 through 32 and 64 to a final 64, allowing progressively more abstract features to be extracted. A Dense layer of 128 neurons with Dropout(0.3) precedes the Softmax output layer for 3-class classification.
Input shape: 128×128×3



Optimizer: Adam (lr=0.0001)

Loss: categorical crossentropy

Regularisation: Dropout(0.3), EarlyStopping(patience=5)



Dataset

~480 images per class were used across three categories: banana, orange, and lemon. An 80/20 training-validation split was applied, with augmentation techniques — rotation, zoom, horizontal flip, brightness variation, and shear — applied exclusively to the training subset.

Results

Validation accuracy of approximately 85% was achieved after training for up to 20 epochs with early stopping. Model weights were saved via ModelCheckpoint at the epoch of highest validation accuracy.

Project Structure

lab5.py - full training and inference pipeline
dataset/ - image data organised by class label
best_model.keras - saved model weights
historia_treningu.png - training and validation curves
wynik_klasyfikacji.png - sample prediction output


Requirements: 

tensorflow, 
opencv-python, 
numpy, 
matplotlib.


Usage

Run training: python lab5.py
To classify a single image, place it as test.jpg in the project root and re-run the script.
