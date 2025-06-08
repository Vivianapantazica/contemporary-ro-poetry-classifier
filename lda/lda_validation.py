import pandas as pd


lda_df = pd.read_csv("../generated/lda_theme_annotated.csv")
manual_df = pd.read_csv("../manual_classified.csv")

# Normalize titles and authors for matching
lda_df["Titlu"] = lda_df["Titlu"].str.strip().str.lower()
lda_df["Autor"] = lda_df["Autor"].str.strip().str.lower()

manual_df["Titlu"] = manual_df["Titlu"].str.strip().str.lower()
manual_df["Autor"] = manual_df["Autor"].str.strip().str.lower()

# Merge both tables on Title and Author
merged_df = pd.merge(lda_df, manual_df, on=["Titlu", "Autor"])
for col in ["Temă", "Tema1", "Tema2", "Tema3"]:
    merged_df[col] = merged_df[col].str.strip().str.lower()

# Unmatched titles and authors
unmatched_lda = lda_df.merge(manual_df, on=["Titlu", "Autor"], how="left", indicator=True)
unmatched_lda = unmatched_lda[unmatched_lda["_merge"] == "left_only"]
print("[INFO] Poezii din LDA fără corespondent în clasificarea manuală:")
print(unmatched_lda[["Titlu", "Autor"]])


# Compute score
def calculate_score(predicted_theme, theme1, theme2, theme3):
    if predicted_theme == theme1 or predicted_theme == theme2 or predicted_theme == theme3:
        return 100
    else:
        return 0


merged_df["Scor"] = merged_df.apply(
    lambda row: calculate_score(
        row["Temă"], row["Tema1"], row["Tema2"], row["Tema3"]
    ),
    axis=1
)

# Compute average score
average_score = merged_df["Scor"].mean()


print("[INFO] Scorurile individuale:")
print(merged_df[["Titlu", "Autor", "Temă", "Tema1", "Tema2", "Tema3", "Scor"]])
print(f"\n[INFO] Scor mediu de clasificare tematică: {average_score:.2f}/100")

merged_df.to_csv("evaluated_lda_accuracy.csv", index=False, encoding="utf-8-sig")
print("[INFO] Scorurile au fost salvate în evaluated_lda_accuracy.csv")
