import pandas as pd

topic_labels = {
    0: "Identity & Otherness",
    1: "Interconnected Structures",
    2: "Philosophical & Ontological Inquiry",
    3: "Self-Referentiality (Ars Poetica)",
    4: "Mundane & Daily Life",
    5: "Eco-Consciousness",
    6: "Digital Realities & Posthumanism",
    7: "Love & Intimacy"
}

df = pd.read_csv("../generated/lda_classified.csv")
df["Temă"] = df["Topic"].map(topic_labels)
df.to_csv("lda_theme_annotated.csv", index=False)
print("Etichetele tematice au fost adăugate cu succes.")
