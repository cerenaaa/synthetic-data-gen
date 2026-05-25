# Synthetic Data Generator

[![CI](https://github.com/cerenaaa/synthetic-data-gen/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/synthetic-data-gen/actions)

GenAI-powered synthetic dataset generation for privacy-safe ML training. Generate realistic tabular data, text corpora, and labeled classification datasets without exposing real customer data.

## Use cases

- Train ML models when real data is scarce or privacy-restricted
- Augment imbalanced datasets (rare fraud, churn, anomaly classes)
- Generate evaluation benchmarks for LLM evals
- Create synthetic PII-free datasets for sharing across teams

## Methods

| Method | Best for |
|---|---|
| `GaussianCopula` | Preserving correlation structure in tabular data |
| `CTGANSynthesizer` | Complex multivariate distributions with mode collapse handling |
| `LLMSynthesizer` | Text data, instruction pairs, labeled NLP corpora |
| `PrivacyAudit` | Measuring re-identification risk of synthetic datasets |

## Quickstart

```bash
pip install -r requirements.txt
python generate.py --type tabular --rows 10000
python generate.py --type text --domain customer_support --n 500
```
