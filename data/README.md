# Dataset Documentation: SmartCache ML Benchmark

## Overview
This directory contains the dataset used for semantic caching evaluation and query complexity routing experiments.

- `raw/queries_raw.json` or `raw/benchmark_dataset.csv`: Initial synthetic and curated queries covering various intent categories and complexity profiles.
- `processed/train_queries.csv`: Training split for the query complexity classifier (no semantic group leakage).
- `processed/val_queries.csv`: Validation split for hyperparameter tuning and threshold calibration.
- `processed/test_queries.csv`: Unseen test split for unbiased evaluation of cache hit rates, false positives, routing accuracy, and cost savings.

## Dataset Fields
| Field | Type | Description |
|---|---|---|
| `query_id` | string | Unique identifier for the question (e.g., `q_001`). |
| `query_text` | string | The actual user question or prompt. |
| `category` | string | Domain category: `general_knowledge`, `simple_factual`, `programming`, `multi_step_reasoning`, `data_analysis`, `long_explanation`, `changing_info`. |
| `complexity_label` | string | `simple` (suitable for small model) or `complex` (requires larger model). |
| `expected_answer` | string | Deterministic reference answer used for quality checks. |
| `paraphrase_group` | string | Cluster identifier (e.g., `grp_014`) grouping semantic variations of identical intent. |
| `is_semantic_match` | bool | Boolean indicating if this query is an intended semantic duplicate of a canonical question in the same group. |
| `expected_route` | string | Recommended inference target: `small` or `large`. |
| `is_cacheable` | bool | Whether the response is deterministic and safe to cache (false for real-time/volatile or personalized queries). |

## Paraphrase Group Design and Leakage Prevention
To evaluate semantic caching realistically:
1. Each group contains a canonical question and 2–4 paraphrases (some exact semantic matches, some subtle negative controls that look similar but differ in crucial details).
2. For routing classifier training, train/val/test splits are partitioned at the **group level** (`paraphrase_group`), ensuring no paraphrases of a test query leak into the training set.

## Limitations of Synthetic & Curated Data
- Synthetic queries may have more uniform syntactic patterns than noisy production traffic.
- Semantic equivalence is labeled based on intent; real users might ask ambiguous or partially overlapping questions.
- Reference answers are deterministic summaries rather than creative generative outputs.
