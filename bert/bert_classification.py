import os
import pandas as pd
from docx import Document
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from transformers import Trainer
import torch.nn as nn
from collections import Counter
import torch
import numpy as np
from transformers import TrainerCallback


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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=weights_tensor.to(model.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


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


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted'),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted')
    }


texts, titles, authors = load_poems(CORPUS_PATH)
labels_df = pd.read_csv(LABELS_CSV)
labels_df["Titlu"] = labels_df["Titlu"].str.strip().str.lower()
labels_df["Autor"] = labels_df["Autor"].str.strip().str.lower()


data = []
for text, title, author in zip(texts, titles, authors):
    row = labels_df[
        (labels_df["Titlu"] == title) &
        (labels_df["Autor"] == author)
    ]
    if not row.empty:
        label = row.iloc[0]["Tema1"]
        data.append({"text": preprocess(text), "label": label})


df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])
print(df["label"].value_counts())

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


df["label_name"] = label_encoder.inverse_transform(df["label"])
df["label_name"].value_counts().plot(kind='barh', title='Distribuția temelor')
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.close()


label_counts = Counter(train_df["label"])
total_samples = sum(label_counts.values())
num_labels = len(label_encoder.classes_)

class_weights = [0] * num_labels
for i in range(num_labels):
    class_weights[i] = total_samples / label_counts.get(i, 1)

weights_tensor = torch.tensor(class_weights, dtype=torch.float)
weights_tensor = weights_tensor / weights_tensor.sum()



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_encoder.classes_)
)


training_args = TrainingArguments(
    seed=42,
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

metric_logger = MetricLoggerCallback()

trainer = WeightedTrainer(
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

trainer.train()
metrics = trainer.evaluate()
print(metrics)


# model.save_pretrained("./finetuned-roberta-poetry")
# tokenizer.save_pretrained("./finetuned-roberta-poetry")

pd.Series(label_encoder.classes_).to_csv("label_map.csv", index=False, header=False)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)


print("[INFO] Evaluez modelul...")
model.eval()
with torch.no_grad():
    preds_logits = []
    true_labels = []

    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds_logits.append(outputs.logits.cpu())
        true_labels.append(labels.cpu())


logits = torch.cat(preds_logits).numpy()
true = torch.cat(true_labels).numpy()
preds = np.argmax(logits, axis=1)


acc = accuracy_score(true, preds)
f1 = f1_score(true, preds, average="weighted")

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nClassification Report:")
print(classification_report(true, preds, target_names=label_encoder.classes_))


conf_mat = confusion_matrix(true, preds)
plt.figure(figsize=(10, 8))
labels = label_encoder.classes_
sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.ylabel('Etichete reale')
plt.xlabel('Etichete prezise')
plt.title('Matrice de confuzie')
plt.tight_layout()
plt.savefig("confusion_matrix_bert2.png")
plt.close()


logs = metric_logger.logs
pd.DataFrame(metric_logger.logs).to_csv("training_logs.csv", index=False)
epochs = [log["epoch"] for log in logs]
f1_scores = [log["f1"] for log in logs]
losses = [log["loss"] for log in logs]

plt.figure()
plt.plot(epochs, f1_scores, label="F1 Score")
plt.plot(epochs, losses, label="Eval Loss")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Evaluation Metrics per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("training_metrics.png")
plt.close()
