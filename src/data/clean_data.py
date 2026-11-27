
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Load original extracted dataset
# --------------------------------------------------

input_path = Path("data/processed/raid_sample.csv")
df = pd.read_csv(input_path)

print("Original records:", len(df))

# --------------------------------------------------
# 2. Basic text cleaning
# --------------------------------------------------

df["text"] = df["text"].fillna("").astype(str).str.strip()

df["word_count"] = df["text"].str.split().str.len()
df["char_count"] = df["text"].str.len()

# --------------------------------------------------
# 3. Identify invalid or very short AI pairs
# --------------------------------------------------

ai = df[df["text_type"].isin(
    ["original_ai", "paraphrased_ai"]
)]

# Identify pairs where either version is below 20 words
short_pair_ids = set(
    ai.loc[ai["word_count"] < 20, "pair_id"]
)

# Identify image placeholders
image_pattern = r"!\[\[|<image>|image_placeholder"

invalid_pair_ids = set(
    ai.loc[
        ai["text"].str.contains(
            image_pattern,
            case=False,
            regex=True,
            na=False
        ),
        "pair_id"
    ]
)

# Remove entire pairs to preserve pairing
pairs_to_remove = short_pair_ids | invalid_pair_ids

print("\n--- AI PAIR EXCLUSIONS ---")
print("Short-text pairs:", len(short_pair_ids))
print("Image-placeholder pairs:", len(invalid_pair_ids))
print("Total unique pairs excluded:", len(pairs_to_remove))

# --------------------------------------------------
# 4. Remove invalid AI pairs
# --------------------------------------------------

cleaned = df[
    ~df["pair_id"].isin(pairs_to_remove)
].copy()

# Apply minimum length to human records
human_short = (
    (cleaned["text_type"] == "human")
    & (cleaned["word_count"] < 20)
)

print("\nShort human records removed:", human_short.sum())

cleaned = cleaned[~human_short].copy()

# --------------------------------------------------
# 5. Remove empty and duplicate records
# --------------------------------------------------

cleaned = cleaned[
    cleaned["text"].str.strip() != ""
].copy()

cleaned = cleaned.drop_duplicates(
    subset=["text"],
    keep="first"
)

# --------------------------------------------------
# 6. Validate final dataset
# --------------------------------------------------

print("\n--- FINAL DATASET VALIDATION ---")

print("Total records:", len(cleaned))
print("\nText categories:")
print(cleaned["text_type"].value_counts())

original_count = (
    cleaned["text_type"] == "original_ai"
).sum()

paraphrase_count = (
    cleaned["text_type"] == "paraphrased_ai"
).sum()

human_count = (
    cleaned["text_type"] == "human"
).sum()

assert original_count == paraphrase_count, (
    "Original and paraphrased counts do not match!"
)

assert cleaned["record_id"].is_unique
assert cleaned["text"].str.strip().ne("").all()

# Every AI pair must contain exactly one original
# and one paraphrase.
pair_counts = cleaned[
    cleaned["text_type"].isin(
        ["original_ai", "paraphrased_ai"]
    )
].groupby(["pair_id", "text_type"]).size().unstack(fill_value=0)

assert (
    (pair_counts["original_ai"] == 1)
    & (pair_counts["paraphrased_ai"] == 1)
).all()

# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

output_path = Path("data/processed/raid_sample_clean.csv")

cleaned.to_csv(output_path, index=False)

print("\n--- CLEANING COMPLETE ---")
print("Original AI:", original_count)
print("Paraphrased AI:", paraphrase_count)
print("Human:", human_count)
print("Total:", len(cleaned))
print("\nSaved:", output_path)

