# SpaCy NER Training on NCBI Disease Dataset

## Overview
This project trains a Named Entity Recognition (NER) model using **SpaCy** on the **NCBI Disease dataset**. The model is trained to recognize disease-related entities.

## Steps Implemented
### 1. Load Dataset
- The dataset is loaded from the Hugging Face repository using:
  ```python
  ds = load_dataset("ncbi/ncbi_disease", trust_remote_code=True)
  ```
- Extracted text and corresponding **NER tags**.

### 2. Data Preprocessing
- Converted dataset into a **pandas DataFrame**.
- Tokenized text using **NLTK**.
- Mapped numeric labels to their respective NER tags.
- Converted data into SpaCy-compatible format **(text, entity spans)**.
- Saved processed data as JSON files (`train_data.json`, `test_data.json`).

### 3. Train a SpaCy NER Model
- Created a **blank English model** using:
  ```python
  nlp = spacy.blank("en")
  ```
- Added a **Named Entity Recognition (NER) pipeline**.
- Added entity labels based on training data.
- Trained for **50 iterations** with minibatch updates.
- Saved the trained model to disk (`trained_spacy_model/`).

### 4. Model Evaluation
- Used the trained model to predict entities on test data.
- Compared **true labels vs. predicted labels**.
- Printed sample predictions for evaluation.
- Generated **classification report** using `sklearn.metrics.classification_report()`.
- Displayed **label distribution statistics**.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install datasets pandas nltk spacy scikit-learn
```

## Running the Code
1. Clone this repository or copy the script.
2. Install dependencies (if not installed already).
3. Run the script:
   ```bash
   python script.py
   ```
4. The trained model will be saved in `trained_spacy_model/`.

## Using the Trained Model
To load and test the trained model:
```python
import spacy
nlp = spacy.load("trained_spacy_model")
text = "Patient was diagnosed with leukemia."
doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])
```

## Results
- The model detects disease-related entities.
- Performance evaluated using **precision, recall, and F1-score**.

## Future Improvements
- Fine-tune the model with more data.
- Experiment with different **dropout rates** and **batch sizes**.
- Use **transformers-based embeddings** for better accuracy.



