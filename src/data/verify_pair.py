
from datasets import load_dataset

TARGET_ID = "0a0495a4-3c8e-4d29-9cfa-cf8644f2d895"

dataset = load_dataset(
    "liamdugan/raid",
    split="train",
    streaming=True
)

print("Searching for the original AI generation...")
print("Target ID:", TARGET_ID)

found = False

for i, row in enumerate(dataset, start=1):

    if row["id"] == TARGET_ID:
        found = True

        print("\nMATCH FOUND!")
        print("-" * 60)
        print("ID:", row["id"])
        print("Model:", row["model"])
        print("Domain:", row["domain"])
        print("Attack:", row["attack"])
        print("Adversarial Source ID:", row["adv_source_id"])
        print("Source ID:", row["source_id"])
        print("\nOriginal text:")
        print(row["generation"][:1000])

        break

    if i % 100_000 == 0:
        print(f"Scanned {i:,} records...")

if not found:
    print("\nNo matching record found in the train split.")
