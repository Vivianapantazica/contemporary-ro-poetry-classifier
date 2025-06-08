import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import joblib
from transformers_interpret import SequenceClassificationExplainer

MODEL_PATH = "./results"
LABEL_ENCODER_PATH = "label_encoder.pkl"
THRESHOLD = 0.5

label_encoder: LabelEncoder = joblib.load(LABEL_ENCODER_PATH)
num_classes = len(label_encoder.classes_)

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess(text):
    text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    text = text.replace("â", "î").replace("Â", "Î")
    return text.lower().strip()


raw_poem = """
îți desenezi viața cu creioane cu vârf tocit
iei pauze lungi și împarți ziua în activități mărunte care să îți țină mâinile ocupate
mintea ta e un labirint cunoscut în care ți-e frică să intri, ca un
test de personalitate pe care-l faci des cu rezultate mereu diferite –
lipsa identității se strecoară printre crăpăturile existenței
mundane: small talk, împărțirea sângelui între tine și pilonii
familiari ai alterității
fugi de confesiuni și te arunci în dezbaterea despre toate lucrurile
care nu te pot atinge
ți-e ușor să ți-l imaginezi pe raskolnikov buimac pe stradă,
neauzindu-și numele
de fapt, ți-e la îndemână să înțelegi șocul descoperirii moralității
cu ochi de spactator omnipotent
(potențele tale le traduci în teoria unei identități pe care nu o deții)
la finalul zilei evadezi din gândurile tale ca dintr-un escape room
– satisfying, dar tu deja știi
cum ai trișat să ajungi
acolo"""

print("Original poem:\n", raw_poem)


preprocessed = preprocess(raw_poem)
print("\n[STEP 1] Preprocessed Text:\n", preprocessed)

inputs = tokenizer(preprocessed, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
inputs = {key: val.to(device) for key, val in inputs.items()}
print("\n[STEP 2] Tokenized input:")
print({k: v.shape for k, v in inputs.items()})

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

print("\n[STEP 3] Raw probabilities:")
for label, prob in zip(label_encoder.classes_, probs):
    print(f"{label:35s}: {prob:.4f}")

predicted_indices = np.where(probs >= THRESHOLD)[0]
predicted_labels = label_encoder.inverse_transform(predicted_indices)

print("\n[STEP 4] Predicted Themes (prob >= 0.5):")
for theme in predicted_labels:
    print(f"- {theme}")

explainer = SequenceClassificationExplainer(
    model=model,
    tokenizer=tokenizer
)

word_attributions = explainer(preprocessed)

print("\n[STEP 5] Explanation per theme:")
for i, (word, score) in enumerate(word_attributions):
    print(f"{word:<15} {score:.4f}")
