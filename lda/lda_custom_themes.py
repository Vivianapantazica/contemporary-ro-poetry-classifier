import pandas as pd

topic_labels = {
    0: "Philosophical & Ontological Inquiry",
    1: "Mundane & Daily Life",
    2: "Identity & Otherness",
    3: "Love & Intimacy",
    4: "Self-Referentiality (Ars Poetica)",
    5: "Interconnected Structures",
    6: "Eco-Consciousness",
    7: "Digital Realities & Posthumanism"
}

df = pd.read_csv("../generated/lda_classified.csv")
df["Temă"] = df["Topic"].map(topic_labels)
df.to_csv("../generated/lda_theme_annotated.csv", index=False)
print("Etichetele tematice au fost adăugate cu succes.")
