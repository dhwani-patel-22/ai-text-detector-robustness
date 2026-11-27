
from datasets import load_dataset
from itertools import islice

# Load RAID using streaming mode
dataset = load_dataset(
    "liamdugan/raid",
    split="train",
    streaming=True
)

print("Dataset loaded successfully!")
print("\nDataset columns:")
print(dataset.features)

# Inspect first 5 records
print("\nInspecting first 5 records:")

for i, row in enumerate(islice(dataset, 5), start=1):
    print(f"\nRecord {i}")
    print("-" * 60)

    for key, value in row.items():
        if key == "generation":
            print(f"{key}: {str(value)[:300]}...")
        else:
            print(f"{key}: {value}")
