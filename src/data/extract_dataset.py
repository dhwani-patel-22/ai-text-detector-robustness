import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset


# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

CONFIG_PATH = Path("configs/dataset_config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

SEED = config["dataset"]["seed"]
SPLIT = config["dataset"]["split"]

TARGET_PAIRS = config["sampling"]["target_ai_pairs"]
TARGET_HUMAN = config["sampling"]["target_human"]

DOMAINS = config["sampling"]["domains"]
MODELS = config["sampling"]["ai_models"]

rng = random.Random(SEED)


# --------------------------------------------------
# 2. Create sampling quotas
# --------------------------------------------------

# 300 pairs across 8 domains x 11 models = 88 groups.
pair_strata = [
    (domain, model)
    for domain in DOMAINS
    for model in MODELS
]

base_quota = TARGET_PAIRS // len(pair_strata)
remainder = TARGET_PAIRS % len(pair_strata)

pair_quotas = {stratum: base_quota for stratum in pair_strata}

# Distribute the remaining samples across randomly
# selected groups, using the fixed random seed.
shuffled_strata = pair_strata.copy()
rng.shuffle(shuffled_strata)

for stratum in shuffled_strata[:remainder]:
    pair_quotas[stratum] += 1


# Distribute human samples across the 8 domains.
human_quotas = {
    domain: TARGET_HUMAN // len(DOMAINS)
    for domain in DOMAINS
}

human_remainder = TARGET_HUMAN % len(DOMAINS)
shuffled_domains = DOMAINS.copy()
rng.shuffle(shuffled_domains)

for domain in shuffled_domains[:human_remainder]:
    human_quotas[domain] += 1


# --------------------------------------------------
# 3. Reservoir sampling helper
# --------------------------------------------------

def reservoir_add(reservoir, count, record, quota):
    """
    Keep a random sample of 'quota' records from a stream
    without loading the complete stream into memory.
    """
    count += 1

    if len(reservoir) < quota:
        reservoir.append(record)
    else:
        index = rng.randrange(count)

        if index < quota:
            reservoir[index] = record

    return count


# --------------------------------------------------
# 4. First pass: sample paraphrases and human records
# --------------------------------------------------

print("\nLoading RAID dataset...")

dataset = load_dataset(
    config["dataset"]["name"],
    split=SPLIT,
    streaming=True
)

paraphrase_samples = defaultdict(list)
paraphrase_counts = defaultdict(int)

human_samples = defaultdict(list)
human_counts = defaultdict(int)

# Prevent selecting multiple paraphrases of the same original.
seen_original_ids = set()

total_scanned = 0

print("\nPASS 1: Sampling paraphrases and human records...\n")

for row in dataset:
    total_scanned += 1

    model = row["model"]
    attack = row["attack"]
    domain = row["domain"]

    if domain not in DOMAINS:
        continue

    # AI paraphrases
    if model in MODELS and attack == "paraphrase":

        original_id = row["adv_source_id"]

        if not original_id or original_id in seen_original_ids:
            continue

        stratum = (domain, model)
        quota = pair_quotas[stratum]

        record = {
            "id": row["id"],
            "adv_source_id": original_id,
            "source_id": row["source_id"],
            "model": model,
            "domain": domain,
            "generation": row["generation"]
        }

        paraphrase_counts[stratum] += 1

        paraphrase_counts[stratum] = reservoir_add(
            paraphrase_samples[stratum],
            paraphrase_counts[stratum] - 1,
            record,
            quota
        )

        seen_original_ids.add(original_id)

    # Clean human passages
    elif model == "human" and attack == "none":

        quota = human_quotas[domain]

        record = {
            "id": row["id"],
            "source_id": row["source_id"],
            "model": model,
            "domain": domain,
            "generation": row["generation"]
        }

        human_counts[domain] = reservoir_add(
            human_samples[domain],
            human_counts[domain],
            record,
            quota
        )

    if total_scanned % 100_000 == 0:
        print(f"Pass 1 scanned {total_scanned:,} records...")


# --------------------------------------------------
# 5. Check whether sampling quotas were reached
# --------------------------------------------------

print("\nChecking sampling quotas...")

sampling_errors = []

for stratum, quota in pair_quotas.items():
    selected = len(paraphrase_samples[stratum])

    if selected < quota:
        sampling_errors.append(
            f"{stratum}: selected {selected}, required {quota}"
        )

for domain, quota in human_quotas.items():
    selected = len(human_samples[domain])

    if selected < quota:
        sampling_errors.append(
            f"Human {domain}: selected {selected}, required {quota}"
        )

if sampling_errors:
    raise RuntimeError(
        "Some sampling quotas were not reached:\n"
        + "\n".join(sampling_errors)
    )


selected_paraphrases = [
    record
    for records in paraphrase_samples.values()
    for record in records
]

selected_humans = [
    record
    for records in human_samples.values()
    for record in records
]

print("\nSampling complete.")
print("Selected paraphrases:", len(selected_paraphrases))
print("Selected human records:", len(selected_humans))


# --------------------------------------------------
# 6. Second pass: retrieve matching original AI texts
# --------------------------------------------------

print("\nPASS 2: Retrieving matching original AI records...\n")

required_original_ids = {
    record["adv_source_id"]
    for record in selected_paraphrases
}

original_records = {}

dataset = load_dataset(
    config["dataset"]["name"],
    split=SPLIT,
    streaming=True
)

total_scanned = 0

for row in dataset:
    total_scanned += 1

    if row["id"] in required_original_ids:
        original_records[row["id"]] = {
            "id": row["id"],
            "source_id": row["source_id"],
            "model": row["model"],
            "domain": row["domain"],
            "generation": row["generation"]
        }

    if len(original_records) == len(required_original_ids):
        break

    if total_scanned % 100_000 == 0:
        print(f"Pass 2 scanned {total_scanned:,} records...")


# --------------------------------------------------
# 7. Validate original-paraphrase pairs
# --------------------------------------------------

print("\nValidating selected pairs...")

validated_pairs = []
validation_errors = []

for paraphrase in selected_paraphrases:

    original_id = paraphrase["adv_source_id"]
    original = original_records.get(original_id)

    if original is None:
        validation_errors.append(
            f"Original not found: {original_id}"
        )
        continue

    if original["model"] != paraphrase["model"]:
        validation_errors.append(
            f"Model mismatch: {original_id}"
        )

    if original["domain"] != paraphrase["domain"]:
        validation_errors.append(
            f"Domain mismatch: {original_id}"
        )

    if original["source_id"] != paraphrase["source_id"]:
        validation_errors.append(
            f"Source ID mismatch: {original_id}"
        )

    validated_pairs.append((original, paraphrase))


if validation_errors:
    raise RuntimeError(
        "Pair validation failed:\n"
        + "\n".join(validation_errors[:20])
    )

if len(validated_pairs) != TARGET_PAIRS:
    raise RuntimeError(
        f"Expected {TARGET_PAIRS} pairs, "
        f"but validated {len(validated_pairs)}."
    )

print("Validated pairs:", len(validated_pairs))
print("Validation errors:", len(validation_errors))


# --------------------------------------------------
# 8. Prepare final 900-record dataset
# --------------------------------------------------

output_rows = []

for original, paraphrase in validated_pairs:

    pair_id = original["id"]

    output_rows.append({
        "record_id": original["id"],
        "pair_id": pair_id,
        "source_id": original["source_id"],
        "model": original["model"],
        "domain": original["domain"],
        "text_type": "original_ai",
        "attack": "none",
        "text": original["generation"]
    })

    output_rows.append({
        "record_id": paraphrase["id"],
        "pair_id": pair_id,
        "source_id": paraphrase["source_id"],
        "model": paraphrase["model"],
        "domain": paraphrase["domain"],
        "text_type": "paraphrased_ai",
        "attack": "paraphrase",
        "text": paraphrase["generation"]
    })


for human in selected_humans:

    output_rows.append({
        "record_id": human["id"],
        "pair_id": "",
        "source_id": human["source_id"],
        "model": "human",
        "domain": human["domain"],
        "text_type": "human",
        "attack": "none",
        "text": human["generation"]
    })


# --------------------------------------------------
# 9. Save CSV and sampling manifest
# --------------------------------------------------

output_dir = Path(config["output"]["directory"])
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "raid_sample.csv"

fieldnames = [
    "record_id",
    "pair_id",
    "source_id",
    "model",
    "domain",
    "text_type",
    "attack",
    "text"
]

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)


manifest = {
    "dataset": config["dataset"]["name"],
    "split": SPLIT,
    "random_seed": SEED,
    "records_in_final_sample": len(output_rows),
    "ai_pairs": len(validated_pairs),
    "original_ai_records": len(validated_pairs),
    "paraphrased_ai_records": len(validated_pairs),
    "human_records": len(selected_humans),
    "domains": DOMAINS,
    "models": MODELS,
    "pair_validation_errors": len(validation_errors),
    "sampling_quotas": {
        "ai_pairs_per_domain_model": {
            f"{domain}|{model}": quota
            for (domain, model), quota in pair_quotas.items()
        },
        "human_per_domain": human_quotas
    }
}

manifest_path = output_dir / "sampling_manifest.json"

with open(manifest_path, "w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=4)


# --------------------------------------------------
# 10. Final summary
# --------------------------------------------------

print("\n" + "=" * 50)
print("DATASET EXTRACTION COMPLETE")
print("=" * 50)

print("Original AI records:", len(validated_pairs))
print("Paraphrased AI records:", len(validated_pairs))
print("Human records:", len(selected_humans))
print("Total records:", len(output_rows))

print("\nSaved files:")
print(csv_path)
print(manifest_path)