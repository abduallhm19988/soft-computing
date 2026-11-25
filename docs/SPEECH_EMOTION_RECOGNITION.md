# Speech Emotion Recognition

## Overview

This project explores speech emotion recognition using audio processing and deep learning. It processes recorded speech, extracts Mel-Frequency Cepstral Coefficients (MFCCs), and trains a neural network to classify each recording into one of eight emotion categories.

The notebook is:

- [`projectdp.ipynb`](../projectdp.ipynb)

## Objective

Predict the emotion represented by a speech recording. The notebook maps the emotion codes in the audio filenames to these classes:

1. Neutral
2. Calm
3. Happy
4. Sad
5. Angry
6. Fear
7. Disgust
8. Surprise

## Audio Data

The notebook expects an audio dataset arranged in folders under a parent directory. It reads the emotion code from the third hyphen-separated field in each filename (the field at index 2 after splitting on `-`).

The notebook currently contains an absolute Windows path from the original development machine. Update that path to the dataset directory on the machine running the notebook. The audio dataset is not included in this repository.

## Workflow

1. **Load audio file paths and labels**  
   Traverse the dataset folders, collect audio file paths, and extract emotion codes from filenames.

2. **Read and inspect audio**  
   Load a 2.4-second segment beginning 0.6 seconds into each recording using Librosa. The notebook also provides waveform and spectrogram visualizations.

3. **Apply audio transformations**  
   The notebook defines functions for adding noise, shifting audio, changing pitch, and stretching time. Its feature-generation routine randomly applies the original, noise, or pitch option before feature extraction.

4. **Extract audio features**  
   Extract MFCCs using `librosa.feature.mfcc`. The feature-generation routine retains arrays with the expected shape of `(20, 104)` and creates multiple feature samples per source recording.

5. **Prepare labels and datasets**  
   Convert emotion labels to one-hot encoded vectors, then split the generated samples into training and test sets using an 80/20 ratio and random seed 11.

6. **Prepare model input**  
   Expand and rearrange feature dimensions to match the neural network's expected input shape.

7. **Build the CNN-LSTM model**  
   The model combines:
   - TimeDistributed 1D convolution and batch normalization
   - Flattening for each time step
   - An LSTM layer with 64 units
   - Dropout layers
   - Fully connected Dense layers
   - An 8-unit Softmax output layer

8. **Train the model**  
   Compile with the Adam optimizer and categorical cross-entropy loss. The notebook configures training for up to 100 epochs with a batch size of 140, using ReduceLROnPlateau and EarlyStopping callbacks.

9. **Evaluate and visualize results**  
   Evaluate the model on the test data, plot training and validation loss/accuracy, and generate predicted labels, a confusion matrix, and a classification report.

10. **Save and reload the model**  
    The notebook includes code to save the trained model in Keras format and load it again for prediction. The save/load paths are specific to the original development environment and should be updated when running elsewhere.

## Technologies

- Python
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Keras

## Running the Notebook

1. Open the notebook with Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the libraries imported by the notebook.
3. Obtain the required audio dataset separately and arrange it in the expected folder structure.
4. Update the dataset path in the notebook.
5. Run the cells in order. Model training may take time depending on the dataset and hardware.

## Results

The notebook contains code to calculate test accuracy and produce training curves, a confusion matrix, and a classification report. Use the outputs from a fresh run to report final performance; this documentation does not add or assume an accuracy value.

## Scope

This is an academic deep-learning experiment for speech emotion classification. The notebook is preserved as originally developed; this documentation update does not change its code. Predictions are model estimates and should not be interpreted as reliable judgments of a person's actual emotional state.
