# Soft Computing Projects

This repository contains two academic Python projects developed as part of my college coursework. The original Jupyter notebooks are preserved; the documentation below explains their goals, workflow, and the techniques used.

## Projects

### 1. Android Malware Detection
**Notebook:** [`soft_poject.ipynb`](./soft_poject.ipynb)

A machine-learning project that classifies Android applications as **Malware** or **Goodware** using the TUANDROMD dataset. The notebook explores feature analysis and preprocessing, trains a linear Support Vector Machine (SVM), and uses a Genetic Algorithm (GA) to search for a subset of features before training and evaluating another SVM model.

- Dataset: TUANDROMD (`TUANDROMD.csv`)
- Main libraries: Python, NumPy, Pandas, Matplotlib, SciPy, scikit-learn
- Techniques: missing-value handling, feature analysis, Min-Max scaling, SVM classification, Genetic Algorithm-based feature selection
- More details: [Android Malware Detection documentation](./docs/ANDROID_MALWARE_DETECTION.md)

### 2. Speech Emotion Recognition
**Notebook:** [`projectdp.ipynb`](./projectdp.ipynb)

A deep-learning project that processes speech recordings and predicts one of eight emotion categories. It extracts audio features using Librosa, applies audio augmentation, and trains a neural network combining time-distributed convolutional layers with an LSTM.

- Emotion classes: Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprise
- Main libraries: Python, Librosa, NumPy, Matplotlib, Seaborn, scikit-learn, Keras
- Techniques: waveform and spectrogram visualization, noise/pitch/time transformations, MFCC feature extraction, CNN-LSTM classification
- More details: [Speech Emotion Recognition documentation](./docs/SPEECH_EMOTION_RECOGNITION.md)

## Repository Structure

```text
soft-computing/
├── projectdp.ipynb
├── soft_poject.ipynb
├── README.md
└── docs/
    ├── ANDROID_MALWARE_DETECTION.md
    └── SPEECH_EMOTION_RECOGNITION.md
```

## Running the Notebooks

Both projects are provided as Jupyter notebooks and can be opened with Jupyter Notebook, JupyterLab, or Google Colab.

1. Clone or download this repository.
2. Install the libraries imported by the notebook you want to run.
3. Obtain the corresponding dataset separately; the datasets are not stored in this repository.
4. Update the dataset path variable in the notebook to match your local dataset location.
5. Run the notebook cells in order.

The notebooks currently contain dataset paths from the original development environment. Those paths are not portable and must point to the dataset location on the machine running the notebook.

## Notes

- The original notebook files and their code have been kept unchanged. The files added in this update are documentation only.
- Dataset files and trained model files are not included.
- Model results should be taken from an actual run of the notebooks with the required datasets and dependencies. No performance figures are claimed in this README.
