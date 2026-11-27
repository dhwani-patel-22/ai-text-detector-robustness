# AI-Text Detector Robustness Under Adversarial Paraphrasing

## 1. Project Overview

**Research question:** Can disagreement between two AI-text detectors serve as an early-warning signal for AI-generated text that becomes difficult to detect after adversarial paraphrasing?

We evaluate two existing detectors:

- **RADAR:** A supervised AI-text detector trained with adversarial learning to improve paraphrase robustness.
- **Fast-DetectGPT:** A zero-shot detector based on conditional probability curvature.

Our goal is not to build another detector. We aim to develop and evaluate a disagreement-based diagnostic framework for identifying potential detection failures.

## 2. Dataset Preparation — Completed

**Dataset:** [RAID — Hugging Face](https://huggingface.co/datasets/liamdugan/raid)

We extracted and cleaned a sample from the RAID training split.

| Text category | Records |
|---|---:|
| Original AI | 298 |
| Paraphrased AI | 298 |
| Human | 300 |
| **Total** | **896** |

Dataset characteristics:
- 298 matched original/paraphrase pairs.
- 11 AI generator models.
- 8 English text domains.
- Each paraphrase is linked to its original using `pair_id`.
- `source_id` identifies the underlying human source.

### Main dataset

`data/processed/raid_sample_clean.csv`

**Use this dataset for all experiments.** Do not independently download, resample, or modify it.

Other useful files:
- `data/processed/sampling_manifest.json` — sampling information.
- `configs/dataset_config.yaml` — dataset configuration.
- `reports/pair_length_analysis.csv` — original vs. paraphrase length analysis.
- `reports/domain_length_summary.csv` — domain-level length analysis.

## 3. Proposed Research Contributions

1. **Detector disagreement:** Investigate whether disagreement between RADAR and Fast-DetectGPT increases on paraphrased AI text and predicts detection failures.
2. **Failure analysis:** Examine vulnerabilities across AI models, domains, and text lengths.
3. **Warning mechanism:** Investigate whether a calibrated disagreement threshold can identify vulnerable AI passages while controlling false alarms on human text.

These are hypotheses to evaluate, not predetermined results.

## 4. Experimental Workflow

1. Prepare source-aware calibration and evaluation splits.
2. Run RADAR on all selected passages.
3. Run Fast-DetectGPT on the same passages.
4. Compare individual detector performance on original and paraphrased AI text.
5. Align detector score directions and scales.
6. Calculate a disagreement score.
7. Evaluate whether disagreement predicts detection failures.
8. Analyze false alarms on human text and performance across different domains and models.

**Important:** Save predictions and continuous scores for every `record_id`, not just binary labels. We need the scores for disagreement analysis.

## 5. Immediate Task: RADAR / Fast-DetectGPT Research and Setup

Please start by investigating the two detectors, focusing on:

- Official GitHub repositories and model checkpoints.
- Installation requirements and dependencies.
- How to run inference on our own text passages.
- What outputs each detector provides (probability, score, or binary prediction).
- Which score direction indicates AI-generated text.
- GPU and computational requirements.
- Any input length limitations or preprocessing requirements.

### Expected deliverable

Create a short document or update the project README with:

| Item | RADAR | Fast-DetectGPT |
|---|---|---|
| GitHub repository | | |
| Model/checkpoint | | |
| Installation requirements | | |
| Input format | | |
| Output score | | |
| Score interpretation | | |
| GPU requirements | | |
| Inference instructions | | |

**Priority:** Determine whether both detectors can be run on our 896-record dataset and identify any setup conflicts before implementation.

Do not start changing the dataset or implementing the disagreement formula yet. We first need to understand both detectors' outputs so we can design a valid comparison.

## 6. Shared Experiment Output

Eventually, each detector should produce a CSV containing:

- `record_id`
- `pair_id`
- `source_id`
- `text_type`
- `model`
- `domain`
- `prediction`
- `score`

These outputs will be merged for the disagreement and robustness analysis.