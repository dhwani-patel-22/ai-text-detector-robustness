
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/raid_sample.csv")

# Calculate word counts
df["word_count"] = df["text"].fillna("").str.split().str.len()

# --------------------------------------------------
# 1. Match original and paraphrased AI records
# --------------------------------------------------

originals = (
    df[df["text_type"] == "original_ai"]
    [["pair_id", "domain", "model", "word_count"]]
    .rename(columns={"word_count": "original_words"})
)

paraphrases = (
    df[df["text_type"] == "paraphrased_ai"]
    [["pair_id", "word_count"]]
    .rename(columns={"word_count": "paraphrased_words"})
)

pairs = originals.merge(paraphrases, on="pair_id", validate="one_to_one")

pairs["word_difference"] = (
    pairs["paraphrased_words"] - pairs["original_words"]
)

pairs["percentage_change"] = (
    pairs["word_difference"] / pairs["original_words"].clip(lower=1)
) * 100

print("=" * 55)
print("PAIRED TEXT LENGTH ANALYSIS")
print("=" * 55)

print("\n--- OVERALL PAIR STATISTICS ---")
print("Number of matched pairs:", len(pairs))
print("Mean original words:", round(pairs["original_words"].mean(), 2))
print("Mean paraphrased words:", round(pairs["paraphrased_words"].mean(), 2))
print("Mean word difference:", round(pairs["word_difference"].mean(), 2))
print("Mean percentage change:", round(pairs["percentage_change"].mean(), 2), "%")

print("\n--- PERCENTAGE CHANGE DISTRIBUTION ---")
print(pairs["percentage_change"].describe().round(2))

print("\nParaphrases shorter than originals:",
      (pairs["word_difference"] < 0).sum())

print("Paraphrases same length:",
      (pairs["word_difference"] == 0).sum())

print("Paraphrases longer than originals:",
      (pairs["word_difference"] > 0).sum())


# --------------------------------------------------
# 2. Compare lengths by domain
# --------------------------------------------------

print("\n--- LENGTH BY DOMAIN ---")

domain_lengths = pairs.groupby("domain").agg(
    original_mean=("original_words", "mean"),
    paraphrase_mean=("paraphrased_words", "mean"),
    mean_difference=("word_difference", "mean"),
    mean_percentage_change=("percentage_change", "mean")
).round(2)

print(domain_lengths)


# --------------------------------------------------
# 3. Investigate short passages
# --------------------------------------------------

print("\n--- SHORT PASSAGES ---")

for category in ["original_ai", "paraphrased_ai", "human"]:
    subset = df[df["text_type"] == category]

    print(f"\n{category}:")
    for threshold in [20, 50]:
        count = (subset["word_count"] < threshold).sum()
        print(f"  Below {threshold} words: {count}")

print("\nAI passages with fewer than 20 words:")

short_ai = df[
    (df["text_type"].isin(["original_ai", "paraphrased_ai"]))
    & (df["word_count"] < 20)
]

print(short_ai[
    ["record_id", "pair_id", "domain", "model", "text_type", "word_count", "text"]
].to_string(index=False))


# --------------------------------------------------
# 4. Investigate long human passages
# --------------------------------------------------

print("\n--- LONG HUMAN PASSAGES ---")

long_human = df[
    (df["text_type"] == "human")
    & (df["word_count"] > 500)
]

print("Human passages above 500 words:", len(long_human))

print(
    long_human[
        ["record_id", "domain", "word_count", "text"]
    ][["record_id", "domain", "word_count"]]
    .to_string(index=False)
)


# --------------------------------------------------
# 5. Save detailed results
# --------------------------------------------------

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

pairs.to_csv(output_dir / "pair_length_analysis.csv", index=False)

domain_lengths.to_csv(output_dir / "domain_length_summary.csv")

print("\nSaved:")
print("reports/pair_length_analysis.csv")
print("reports/domain_length_summary.csv")
