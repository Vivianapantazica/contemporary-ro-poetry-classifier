import re
import spacy
import os
from docx import Document
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import multiprocessing
import pandas as pd

nlp = spacy.load("ro_core_news_sm")


with open("../stopwords_ro.txt", encoding="utf-8") as f:
    stopwords_ro = set(f.read().splitlines())
    custom_stopwords = {
        "vrea", "vedea", "veni", "merge", "simți", "zice", "face", "avea",
        "trece", "rămâne", "spune", "ajunge", "pleca", "părea", "sta", "lua",
        "scrie", "vorbi", "aduce", "duce", "arăta", "începe", "întoarce"
    }
    stopwords_ro = stopwords_ro.union(custom_stopwords)


def clean_text(text):
    text = re.sub(r'[^a-zA-ZăîâșțĂÎÂȘȚ ]+', ' ', text)
    text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    text = text.replace("â", "î").replace("Â", "Î")
    return text


def preprocess(text):
    text = clean_text(text.lower())
    doc = nlp(text)

    return [
        token.lemma_ for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.lemma_ not in stopwords_ro
        and len(token.lemma_) > 2
    ]


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def load_poems(corpus_path):
    texts = []
    titles = []
    authors = []

    for author in os.listdir(corpus_path):
        author_path = os.path.join(corpus_path, author)
        if not os.path.isdir(author_path):
            continue

        for poem_file in os.listdir(author_path):
            if not poem_file.lower().endswith('.docx'):
                continue

            poem_path = os.path.join(author_path, poem_file)
            try:
                text = read_docx(poem_path).strip()
                texts.append(text)
                titles.append(os.path.splitext(poem_file)[0])
                authors.append(author)
            except Exception as e:
                print(f"Could not read {poem_path}: {e}")

    return texts, titles, authors


corpus_path = "../romanian-poetry-corpus"
texts, titles, authors = load_poems(corpus_path)

print("[INFO] Se preprocesează textele...")
processed_texts = [preprocess(poem) for poem in texts]
bigram = Phrases(processed_texts, min_count=2, threshold=5)
bigram_mod = Phraser(bigram)
processed_texts = [bigram_mod[doc] for doc in processed_texts]
print(f"[INFO] Preprocesare finalizată pentru {len(processed_texts)} poezii.")

print("[INFO] Se construiește dicționarul și corpusul...")
dictionary = corpora.Dictionary(processed_texts)
dictionary.filter_extremes(no_below=5, no_above=0.4, keep_n=10000)
dictionary.filter_tokens(bad_ids=[dictionary.token2id[token] for token in dictionary.token2id if len(token) <= 2])
corpus = [dictionary.doc2bow(text) for text in processed_texts]
print(f"[INFO] Dicționar creat cu {len(dictionary)} termeni.")
print(f"[INFO] Corpus creat cu {len(corpus)} documente.")

print("[INFO] Se antrenează modelul LDA...")
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=8,
    random_state=42,
    passes=20,
    iterations=100,
    alpha='auto',
    eta='auto',
    per_word_topics=True
)
print("[INFO] Antrenare LDA finalizată.")
# lda_model.save("lda_model_improved")


def compute_coherence():
    coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"[INFO] Coerența modelului: {coherence_score:.4f}")


print("[INFO] Clustere de cuvinte cheie pentru fiecare temă:")
for idx, topic in lda_model.print_topics(-1):
    print(f"  • Tema #{idx}: {topic}")


def get_document_topics(lda_model, corpus):
    doc_topics = []
    for i, bow in enumerate(corpus):
        topic_probs = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_probs, key=lambda x: x[1])
        doc_topics.append(dominant_topic[0])
    return doc_topics


print("[INFO] Se determină tema dominantă pentru fiecare poezie...")
poem_topics = get_document_topics(lda_model, corpus)

results = []
for i in range(len(texts)):
    lda_topic = poem_topics[i]
    results.append({
        "Titlu": titles[i],
        "Autor": authors[i],
        "Topic": lda_topic
    })


df = pd.DataFrame(results)
output_path = "../generated/lda_classified.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"[INFO] Rezultatele au fost salvate în: {output_path}")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    with open("../generated/lda_topics.txt", "w", encoding="utf-8") as f:
        for idx, topic in lda_model.print_topics(-1):
            f.write(f"Tema #{idx}: {topic}\n")
    compute_coherence()