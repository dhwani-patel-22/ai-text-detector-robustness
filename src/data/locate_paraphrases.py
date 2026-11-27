
from datasets import load_dataset

dataset = load_dataset(
    "liamdugan/raid",
    split="train",
    streaming=True
)

scanned = 0
found = 0
target = 10

print("Searching for paraphrased AI records...\n")

for row in dataset:
    scanned += 1

    if row["model"] != "human" and row["attack"] == "paraphrase":
        found += 1

        print(f"\nPARAPHRASE RECORD {found}")
        print("-" * 60)
        print("ID:", row["id"])
        print("Source ID:", row["source_id"])
        print("Adversarial Source ID:", row["adv_source_id"])
        print("Model:", row["model"])
        print("Domain:", row["domain"])
        print("Attack:", row["attack"])
        print("Text:", row["generation"][:400])

        if found >= target:
            break

    if scanned % 100_000 == 0:
        print(f"Scanned {scanned:,} records; found {found} paraphrases.")

print("\nSEARCH COMPLETE")
print("Total records scanned:", f"{scanned:,}")
print("Paraphrases found:", found)
