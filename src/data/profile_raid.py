
from datasets import load_dataset
from collections import Counter

dataset = load_dataset(
    "liamdugan/raid",
    split="train",
    streaming=True
)

total = 0

model_counts = Counter()
attack_counts = Counter()
domain_counts = Counter()

# More useful combinations for our experiment
domain_attack_counts = Counter()
model_attack_counts = Counter()
domain_model_attack_counts = Counter()

print("Profiling the complete RAID training split...\n")

for row in dataset:
    total += 1

    model = row["model"]
    attack = row["attack"]
    domain = row["domain"]

    model_counts[model] += 1
    attack_counts[attack] += 1
    domain_counts[domain] += 1

    domain_attack_counts[(domain, attack)] += 1
    model_attack_counts[(model, attack)] += 1
    domain_model_attack_counts[(domain, model, attack)] += 1

    if total % 100_000 == 0:
        print(f"Scanned {total:,} records...")

# Display profile
print("\n" + "=" * 60)
print("RAID DATASET PROFILE")
print("=" * 60)

print(f"\nTotal records: {total:,}")

print("\n--- MODELS ---")
for key, value in model_counts.most_common():
    print(f"{key}: {value:,}")

print("\n--- ATTACK TYPES ---")
for key, value in attack_counts.most_common():
    print(f"{key}: {value:,}")

print("\n--- DOMAINS ---")
for key, value in domain_counts.most_common():
    print(f"{key}: {value:,}")

print("\n--- DOMAIN + ATTACK ---")
for key, value in sorted(domain_attack_counts.items()):
    print(f"{key}: {value:,}")

print("\n--- MODEL + ATTACK ---")
for key, value in sorted(model_attack_counts.items()):
    print(f"{key}: {value:,}")

# Save a copy of the profile
with open("reports/raid_profile.txt", "w") as f:
    f.write(f"Total records: {total:,}\n\n")

    for title, counter in [
        ("MODELS", model_counts),
        ("ATTACKS", attack_counts),
        ("DOMAINS", domain_counts),
        ("DOMAIN + ATTACK", domain_attack_counts),
        ("MODEL + ATTACK", model_attack_counts),
    ]:
        f.write(f"\n--- {title} ---\n")
        for key, value in sorted(counter.items()):
            f.write(f"{key}: {value:,}\n")

print("\nProfile saved to reports/raid_profile.txt")
