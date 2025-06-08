import os
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from docx import Document
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
)
import torch.nn as nn


class MetricLoggerCallback(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logs.append({
                "epoch": state.epoch,
                "f1": metrics.get("eval_f1"),
                "accuracy": metrics.get("eval_accuracy"),
                "loss": metrics.get("eval_loss")
            })


# MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"
MODEL_NAME = "readerbench/RoBERT-base"
# MODEL_NAME = "xlm-roberta-base"
CORPUS_PATH = "../romanian-poetry-corpus"
LABELS_CSV = "../manual_classified.csv"
NUM_EPOCHS = 10
BATCH_SIZE = 8


def preprocess(text):
    text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    text = text.replace("â", "î").replace("Â", "Î")
    return text.lower()


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def load_poems(corpus_path):
    texts, titles, authors = [], [], []
    for author in os.listdir(corpus_path):
        author_path = os.path.join(corpus_path, author)
        if not os.path.isdir(author_path):
            continue
        for poem_file in os.listdir(author_path):
            if not poem_file.lower().endswith(".docx"):
                continue
            try:
                poem_path = os.path.join(author_path, poem_file)
                text = read_docx(poem_path).strip()
                texts.append(text)
                titles.append(os.path.splitext(poem_file)[0].strip().lower())
                authors.append(author.strip().lower())
            except Exception as e:
                print(f"Could not read {poem_path}: {e}")
    return texts, titles, authors


texts, titles, authors = load_poems(CORPUS_PATH)
labels_df = pd.read_csv(LABELS_CSV)
labels_df["Titlu"] = labels_df["Titlu"].str.strip().str.lower()
labels_df["Autor"] = labels_df["Autor"].str.strip().str.lower()

# Multi-label preprocessing
all_labels = pd.unique(labels_df[["Tema1", "Tema2", "Tema3"]].values.ravel())
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
num_classes = len(label_encoder.classes_)


def encode_labels(row):
    labels = set()
    for col in ["Tema1", "Tema2", "Tema3"]:
        val = row[col]
        if pd.notna(val):
            labels.add(val.strip())
    multi_hot = np.zeros(num_classes)
    encoded = label_encoder.transform(list(labels))
    multi_hot[encoded] = 1
    return multi_hot

examples = []
for text, title, author in zip(texts, titles, authors):
    row = labels_df[(labels_df["Titlu"] == title) & (labels_df["Autor"] == author)]
    if not row.empty:
        encoded = encode_labels(row.iloc[0])
        examples.append({"text": preprocess(text), "labels": encoded})


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df = pd.DataFrame(examples)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


def tokenize(example):
    encoded = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    encoded["labels"] = example["labels"]
    return encoded


train_dataset = Dataset.from_pandas(train_df).map(tokenize)
test_dataset = Dataset.from_pandas(test_df).map(tokenize)


class WeightedMultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"].float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes, problem_type="multi_label_classification"
)


def compute_metrics(pred):
    probs = torch.sigmoid(torch.tensor(pred.predictions))
    preds = (probs > 0.5).int().numpy()
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0)
    }


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

metric_logger = MetricLoggerCallback()
trainer = WeightedMultilabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        metric_logger
    ]
)
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

trainer.train()
trainer.save_model("./results")  # Or a named folder of your choice
metrics = trainer.evaluate()
print(metrics)


print("[INFO] Generating detailed metrics and confusion matrix...")
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits)
        preds = (probs > 0.5).int()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())


y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_labels).numpy()

f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
print(f"\n[RESULT] Weighted F1 Score: {f1:.4f}")
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))

fig, axes = plt.subplots(nrows=int(np.ceil(num_classes / 3)), ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i in range(num_classes):
    cm = confusion_matrix(y_true[:, i], y_pred[:, i])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{label_encoder.classes_[i]}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")


for j in range(num_classes, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
