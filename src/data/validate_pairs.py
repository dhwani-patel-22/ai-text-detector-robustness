
from datasets import load_dataset
from collections import Counter

TARGET_PARAPHRASES = 100

dataset = load_dataset(
    "liamdugan/raid",
    split="train",
    streaming=True
)

# Store original AI records by their ID
originals = {}

# Store validation statistics
paraphrase_count = 0
matched_count = 0
unmatched_count = 0
mismatch_count = 0

paraphrases_per_original = Counter()

for row in dataset:

    model = row["model"]
    attack = row["attack"]

    # Store original AI generations
    if model != "human" and attack == "none":
        originals[row["id"]] = {
            "model": model,
            "domain": row["domain"],
            "source_id": row["source_id"]
        }

    # Validate paraphrased AI records
    elif model != "human" and attack == "paraphrase":

        paraphrase_count += 1

        source_id = row["adv_source_id"]
        original = originals.get(source_id)

        if original is None:
            unmatched_count += 1
            print("\nUNMATCHED PARAPHRASE:", row["id"])
        else:
            matched_count += 1
            paraphrases_per_original[source_id] += 1

            # Check consistency
            if (
                original["model"] != row["model"]
                or original["domain"] != row["domain"]
                or original["source_id"] != row["source_id"]
            ):
                mismatch_count += 1
                print("\nMETADATA MISMATCH:", row["id"])

        if paraphrase_count >= TARGET_PARAPHRASES:
            break

# Results
print("\n" + "=" * 50)
print("PAIR VALIDATION RESULTS")
print("=" * 50)

print("Paraphrases inspected:", paraphrase_count)
print("Matched originals:", matched_count)
print("Unmatched paraphrases:", unmatched_count)
print("Metadata mismatches:", mismatch_count)

print(
    "Unique original generations:",
    len(paraphrases_per_original)
)

print(
    "Originals with multiple paraphrases:",
    sum(1 for count in paraphrases_per_original.values() if count > 1)
)

print("\nParaphrases per original:")
print(paraphrases_per_original.most_common(10))
