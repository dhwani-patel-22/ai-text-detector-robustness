
from datasets import load_dataset
from collections import Counter

# Load RAID in streaming mode
dataset = load_dataset(
    "liamdugan/raid",
    split="train",
    streaming=True
)

# Shuffle the stream for a more varied exploratory sample
dataset = dataset.shuffle(seed=42, buffer_size=10_000)

# Counters for dataset exploration
model_counts = Counter()
attack_counts = Counter()
domain_counts = Counter()
combination_counts = Counter()

original_ids = set()
paraphrase_source_ids = set()

human_count = 0
ai_count = 0
total_records = 0

SAMPLE_SIZE = 20_000

for row in dataset:
    total_records += 1

    model = row.get("model")
    attack = row.get("attack")
    domain = row.get("domain")

    model_counts[model] += 1
    attack_counts[attack] += 1
    domain_counts[domain] += 1

    combination_counts[(model, attack)] += 1

    if model == "human":
        human_count += 1
    else:
        ai_count += 1

    # Track original AI generations
    if model != "human" and attack == "none":
        original_ids.add(row.get("id"))

    # Track paraphrased AI generations
    if model != "human" and attack == "paraphrase":
        paraphrase_source_ids.add(row.get("adv_source_id"))

    if total_records >= SAMPLE_SIZE:
        break

# Display results
print(f"\nTotal records inspected: {total_records}")

print("\n1. HUMAN VS AI")
print("Human:", human_count)
print("AI:", ai_count)

print("\n2. MODELS")
for key, value in model_counts.most_common():
    print(f"{key}: {value}")

print("\n3. ATTACK TYPES")
for key, value in attack_counts.most_common():
    print(f"{key}: {value}")

print("\n4. DOMAINS")
for key, value in domain_counts.most_common():
    print(f"{key}: {value}")

print("\n5. MODEL AND ATTACK COMBINATIONS")
for key, value in combination_counts.most_common():
    print(f"{key}: {value}")

print("\n6. ORIGINAL-PARAPHRASE ID MATCHES")
matches = original_ids.intersection(paraphrase_source_ids)
print("Original IDs found:", len(original_ids))
print("Paraphrase source IDs found:", len(paraphrase_source_ids))
print("Matching IDs in sample:", len(matches))
