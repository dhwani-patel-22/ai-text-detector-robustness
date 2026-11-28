
# Predicting AI-Text Detector Failure Under Adversarial Paraphrasing
### A Feature-Drift and Detector-Disagreement Analysis

## 1. Problem and Motivation

AI-generated text detectors are increasingly used to assess content provenance in education, online information, and other settings. Yet these tools identify patterns associated with machine-generated writing rather than authorship itself.

Adversarial paraphrasing makes this weakness especially important: an instruction-following language model can rewrite AI-generated text to lower a detector’s AI score while preserving much of the original meaning. Such failures can undermine reliability, while false accusations against human writers make careful evaluation essential.

Cheng et al. (2025) demonstrated that detector-guided paraphrasing can transfer across different detection systems. Our project therefore asks not only whether detectors fail, but which measurable signals help explain or anticipate particular failures.

## 2. Previous Work and Research Gap

Existing studies address different parts of this problem:

- **RADAR** uses adversarial training to improve robustness to paraphrasing (Hu et al., 2023).
- **Fast-DetectGPT** detects machine-generated text using a probability-based, zero-shot signal (Bao et al., 2024).
- **Cheng et al. (2025)** evaluated a strong detector-guided paraphrasing attack and showed substantial losses in detection performance.

These contributions establish both the attack’s effectiveness and the value of more robust detectors. However, aggregate detection scores alone do not explain why a particular passage changes from "AI" to "human," or whether inexpensive signals can identify vulnerable cases.

We will study interpretable text-feature shifts alongside disagreement between two different detector families, without claiming to introduce a new attack or state-of-the-art detector.

## 3. Project Objectives

Our four objectives are to:

1. Quantify detection performance before and after adversarial paraphrasing.
2. Identify which statistical, stylistic, and semantic changes are associated with AI-to-human prediction flips.
3. Determine whether supervised and zero-shot detectors fail on the same texts.
4. Evaluate whether disagreement between detectors can act as a simple warning signal for paraphrased AI text.

## 4. Proposed Methodology

### 4.1 Data and Attack

We will select a manageable English-language sample from a public AI-versus-human corpus such as **MAGE** (Li et al., 2024), starting with approximately:

| Text Category | Initial Sample Size |
|---|---:|
| AI-generated passages | 300 |
| Human-written passages | 300 |
| Paraphrased AI passages | 300 |
| **Total** | **900** |

We will:

- Control for text length and topic where feasible.
- Record the generator used for AI passages.
- Use a public implementation of detector-guided adversarial paraphrasing.
- Create one rewritten version of each AI passage.
- Retain original–paraphrase pairs for feature comparison.
- Use human passages as a negative comparison set.

If computing resources are limited, we will reduce the sample size while retaining the same evaluation design.

### 4.2 Detectors and Evaluation

We will compare two different detector families:

| Detector | Type | Main Approach |
|---|---|---|
| RADAR | Supervised | Adversarially trained AI-text detector |
| Fast-DetectGPT | Statistical / Zero-shot | Conditional probability curvature |

Whenever feasible, attack guidance will use a separate fixed scoring model so the evaluation detectors are not directly optimized against.

Each detector will score:

1. Original AI-generated text.
2. Adversarially paraphrased AI text.
3. Human-written text.

**Evaluation Metrics**

We will report:

- AUROC
- Accuracy
- F1-score
- True-positive rate at a 1% false-positive rate

A human-text calibration subset will be used to set thresholds.

For passages correctly identified as AI before rewriting, we will measure the fraction whose predicted label flips to human afterward.

### 4.3 Feature and Disagreement Analysis

For each original and paraphrase pair, we will compute changes in the following features:

| Feature | Purpose |
|---|---|
| Perplexity | Measures text predictability |
| Average sentence length | Captures sentence structure |
| Sentence-length variation (burstiness) | Measures variation in writing rhythm |
| Function-word frequency | Captures stylistic patterns |
| Part-of-speech distributions | Examines grammatical patterns |
| SBERT cosine similarity | Estimates semantic similarity |

Sentence-BERT (SBERT) cosine similarity will measure approximate meaning preservation.

We will compare feature shifts for flipped versus non-flipped cases and use a simple held-out logistic-regression analysis to test whether these shifts predict label flips.

**Detector Disagreement Analysis**

Separately, we will:

- Calculate how frequently RADAR and Fast-DetectGPT disagree.
- Compare disagreement patterns across original AI, paraphrased AI, and human text.
- Test whether disagreement helps distinguish paraphrased from unmodified text.

We will explicitly distinguish:

- **Pair-based feature drift:** Requires access to both the original and paraphrased passage.
- **Single-text detector disagreement:** Requires only the detector predictions for an individual passage.

This distinction is important because detector disagreement could potentially provide a warning signal without requiring access to the original text.

## 5. Expected Outcomes

We expect paraphrasing to reduce detection performance, potentially by different amounts for the two detector types.

The main deliverables will include:

- A reproducible dataset of original and paraphrased text pairs.
- Before-and-after detector performance tables.
- Plots relating feature drift to AI-to-human prediction flips.
- A comparison of RADAR and Fast-DetectGPT disagreement.
- An exploratory analysis of whether disagreement can act as a warning signal.
- A final research report and reproducible code.

## 6. Limitations

The study has several limitations:

- A relatively small sample may limit generalizability.
- Only two selected detectors will be evaluated.
- Only one adversarial paraphrasing method will be considered.
- SBERT similarity cannot guarantee factual or semantic equivalence.
- Feature changes associated with detector failure do not necessarily establish causation.

Any warning signal will be presented as an exploratory research finding, not a basis for making definitive authorship judgments.

## 7. Work Plan

| Phase | Activities |
|---|---|
| Phase 1: Data Preparation | Prepare the dataset, generate adversarial paraphrases, and collect detector outputs. |
| Phase 2: Analysis | Calculate text features, evaluate detector performance, analyze feature drift and detector disagreement, and write the final report. |

## 8. References

1. Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y. (2024). *Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature.* ICLR.

2. Cheng, Y., Sadasivan, V. S., Saberi, M., Saha, S., & Feizi, S. (2025). *Adversarial paraphrasing: A universal attack for humanizing AI-generated text.* NeurIPS 38.

3. Hu, X., Chen, P.-Y., & Ho, T.-Y. (2023). *RADAR: Robust AI-text detection via adversarial learning.* NeurIPS 36.

4. Li, Y., Li, Q., Cui, L., et al. (2024). *MAGE: Machine-generated text detection in the wild.* ACL 2024, 36–53.
