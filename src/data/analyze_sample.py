
import pandas as pd
from pathlib import Path

# Load extracted dataset
file_path = Path("data/processed/raid_sample.csv")
df = pd.read_csv(file_path)

print("=" * 55)
print("EXTRACTED DATASET QUALITY ANALYSIS")
print("=" * 55)

# 1. Basic information
print("\n--- DATASET OVERVIEW ---")
print("Total records:", len(df))
print("Columns:", df.columns.tolist())
print("\nText categories:")
print(df["text_type"].value_counts())

# 2. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 3. Empty text
df["text"] = df["text"].fillna("")
df["word_count"] = df["text"].str.split().str.len()
df["char_count"] = df["text"].str.len()

print("\nEmpty texts:", (df["text"].str.strip() == "").sum())

# 4. Exact duplicates
print("\n--- DUPLICATE ANALYSIS ---")
print("Duplicate record IDs:", df["record_id"].duplicated().sum())
print("Duplicate text records:", df["text"].duplicated().sum())

# 5. Text length statistics
print("\n--- TEXT LENGTH STATISTICS ---")
print(
    df.groupby("text_type")["word_count"]
    .describe()
    .round(2)
)

# 6. Domain distribution
print("\n--- DOMAIN DISTRIBUTION ---")
print(pd.crosstab(df["domain"], df["text_type"]))

# 7. AI model distribution
print("\n--- AI MODEL DISTRIBUTION ---")
print(
    df[df["text_type"] != "human"]["model"]
    .value_counts()
)

# 8. Original vs paraphrased length comparison
print("\n--- ORIGINAL VS PARAPHRASED ---")
ai = df[df["text_type"] != "human"]

print(
    ai.groupby("text_type")[
        ["word_count", "char_count"]
    ].mean().round(2)
)

# 9. Save detailed report
report_path = Path("reports/dataset_quality_report.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("DATASET QUALITY REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total records: {len(df)}\n\n")
    f.write("Missing values:\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\nText length statistics:\n")
    f.write(
        df.groupby("text_type")["word_count"]
        .describe().round(2).to_string()
    )
    f.write("\n\nDomain distribution:\n")
    f.write(pd.crosstab(df["domain"], df["text_type"]).to_string())

print("\nReport saved to:", report_path)
