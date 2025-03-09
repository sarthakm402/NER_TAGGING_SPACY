from datasets import load_dataset
import pandas as pd
from nltk.tokenize import word_tokenize
import json
import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from sklearn.metrics import classification_report
from collections import Counter

# Step 1: Load the dataset
ds = load_dataset("ncbi/ncbi_disease", trust_remote_code=True)
label_names = ds["train"].features["ner_tags"].feature.names  # ['O', 'B-Disease', 'I-Disease']
print("NER Labels:", label_names)

# Step 2: Convert dataset into DataFrame
def convert_to_dataframe(dataset):
    return pd.DataFrame({
        "id": dataset["id"],
        "text": [" ".join(tokens) for tokens in dataset["tokens"]],
        "ner_tags": dataset["ner_tags"]
    })

df_train = convert_to_dataframe(ds["train"])
df_test = convert_to_dataframe(ds["test"])

df_train["tokens"] = df_train["text"].apply(word_tokenize)
df_test["tokens"] = df_test["text"].apply(word_tokenize)

df_train["new_ner_tags"] = df_train["ner_tags"].apply(lambda tags: [label_names[i] for i in tags])
df_test["new_ner_tags"] = df_test["ner_tags"].apply(lambda tags: [label_names[i] for i in tags])

# Step 3: Convert data to spaCy format
def convert_to_spacy_format(text, tokens, ner_tags):
    entities = []
    char_start = 0
    
    for token, tag in zip(tokens, ner_tags):
        char_start = text.find(token, char_start)
        if char_start == -1:
            continue
        char_end = char_start + len(token)

        if tag != "O":
            entities.append((char_start, char_end, tag))
        char_start = char_end  
    
    return (text, {"entities": entities})

train_data_spacy = df_train.apply(lambda row: convert_to_spacy_format(row["text"], row["tokens"], row["new_ner_tags"]), axis=1).tolist()
test_data_spacy = df_test.apply(lambda row: convert_to_spacy_format(row["text"], row["tokens"], row["new_ner_tags"]), axis=1).tolist()

# Save data as JSON
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data_spacy, f, indent=4, ensure_ascii=False)
with open("test_data.json", "w", encoding="utf-8") as f1:
    json.dump(test_data_spacy, f1, indent=4, ensure_ascii=False)

# Step 4: Train spaCy Model
nlp = spacy.blank("en")  # Create a blank English model
ner = nlp.add_pipe("ner", last=True)

for _, annotations in train_data_spacy:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "ner"]):
    optimizer = nlp.begin_training()
    for i in range(50):  # Training for 30 iterations
        losses = {}
        batches = minibatch(train_data_spacy, size=8)
        for batch in batches:
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.2, losses=losses)
        print(f"Iteration {i+1}, Losses: {losses}")

# Save trained model
nlp.to_disk("trained_spacy_model")
nlp = spacy.load("trained_spacy_model")

# Step 5: Evaluate Model
def get_predicted_labels(text):
    doc = nlp(text)
    tokens = text.split()
    labels = ["O"] * len(tokens)   # Default label is "O"
    
    for ent in doc.ents:
        ent_tokens = ent.text.split()
        start_idx = next((i for i, token in enumerate(tokens) if token == ent_tokens[0]), None)
        if start_idx is not None:
            labels[start_idx] = f"{ent.label_}"
            for i in range(1, len(ent_tokens)):
                if start_idx + i < len(tokens):
                    labels[start_idx + i] = f"{ent.label_}"
    
    return labels

true_labels, pred_labels = [], []

# Print sample predictions
print("\n🔹 Sample Model Predictions:")
for i, (text, annotations) in enumerate(test_data_spacy[:5]):
    tokens = text.split()
    true_label_seq = ["O"] * len(tokens)

    for start, end, label in annotations["entities"]:
        entity_tokens = text[start:end].split()
        start_idx = next((i for i, token in enumerate(tokens) if token == entity_tokens[0]), None)
        if start_idx is not None:
            true_label_seq[start_idx] = f"{label}"
            for i in range(1, len(entity_tokens)):
                if start_idx + i < len(tokens):
                    true_label_seq[start_idx + i] = f"{label}"
    
    pred_label_seq = get_predicted_labels(text)
    true_labels.extend(true_label_seq)
    pred_labels.extend(pred_label_seq)
    
    print(f"\n🔹 Example {i+1}:")
    print("Text: ", text)
    print("True: ", true_label_seq)
    print("Pred: ", pred_label_seq)

# Model Evaluation
print("\n--- Model Evaluation ---")
print(classification_report(true_labels, pred_labels, labels=["B-Disease", "I-Disease", "O"]))

# Label distributions
print("\nTrue Label Distribution:", Counter(true_labels))
print("Predicted Label Distribution:", Counter(pred_labels))





