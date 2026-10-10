"""
Script to generate the 6 comprehensive, executable Jupyter Notebooks for SmartCache ML.
Uses nbformat to construct well-structured .ipynb files.
"""

import nbformat as nbf
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook_01():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown intro
    cells.append(nbf.v4.new_markdown_cell("""# Notebook 1: Project Setup and Baseline
## SmartCache ML: Semantic Caching and Cost-Aware LLM Routing

### 1. Problem Overview
Language model inference is computationally expensive, has non-trivial latency (often hundreds of milliseconds to multiple seconds), and costs money per input/output token when using cloud APIs. In many production systems (e.g. customer support, enterprise internal search, FAQs, developer docs), users frequently ask identical or semantically equivalent questions.

Repeatedly forwarding identical or near-identical prompts to large language models wastes compute and budget. In this notebook, we establish our project baseline:
1. We inspect our benchmark dataset.
2. We establish a **No-Cache Baseline** where every query triggers full model inference.
3. We build an **Exact-Match Cache** and observe why it fails on realistic user traffic.
4. We formalize our evaluation metrics: latency (p50, p95), inference cost (USD), cache hit rate, and answer quality.

---
### Learning Objectives
By the end of this notebook, you will be able to:
- Formulate the cost and latency bottlenecks of un-cached LLM systems.
- Explain the mechanics of exact-match string caching and its limitations.
- Define reproducible evaluation metrics: mean, p50, and p95 latency, financial cost estimations, and token overlap quality.
- Articulate the baseline performance numbers for technical interview discussions."""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from smartcache.config import settings
from smartcache.llm_provider import MockProvider
from smartcache.cache import ExactMatchCache
from smartcache.cost_tracker import CostTracker, RequestRecord
from smartcache.metrics import calculate_token_overlap_f1

print("Environment and SmartCache modules loaded successfully!")
print(f"Data raw path: {settings.data_raw_path}")"""))

    # Data Loading
    cells.append(nbf.v4.new_markdown_cell("""### 2. Loading and Inspecting the Benchmark Dataset
Let's load the benchmark queries generated for this project and examine class balances, categories, and paraphrase distributions."""))

    cells.append(nbf.v4.new_code_cell("""dataset_path = settings.data_raw_path / "benchmark_dataset.csv"
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_sample_data.py first.")

df = pd.read_csv(dataset_path)
print(f"Total queries: {len(df)}")
print(f"Unique paraphrase groups: {df['paraphrase_group'].nunique()}")
print("\\nClass distribution:")
print(df['complexity_label'].value_counts())
print("\\nCategory distribution:")
print(df['category'].value_counts())
df.head(6)"""))

    # Data visualization
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df['complexity_label'].value_counts().plot(kind='bar', color=['#2b5c8f', '#d95f02'])
plt.title("Query Complexity Distribution")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
df['category'].value_counts().plot(kind='barh', color='#7570b3')
plt.title("Queries by Category")
plt.xlabel("Count")
plt.tight_layout()
plt.show()"""))

    # No cache baseline
    cells.append(nbf.v4.new_markdown_cell("""### 3. Establishing the No-Cache Baseline
In the no-cache baseline, every incoming query is dispatched directly to our LLM provider.
We measure total latency, estimated token counts, and cost."""))

    cells.append(nbf.v4.new_code_cell("""# Instantiate MockProvider with deterministic answers
provider = MockProvider()
tracker_baseline = CostTracker()

# Simulate serving the entire query workload without caching (always using large model as baseline standard)
print("Running No-Cache Baseline over test queries...")
sample_queries = df.head(30)

for _, row in sample_queries.iterrows():
    t_start = time.perf_counter()
    resp = provider.generate(row['query_text'], model_type="large")
    latency = time.perf_counter() - t_start
    
    cost = tracker_baseline.calculate_cost(
        route="large",
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        is_cache_hit=False
    )
    
    rec = RequestRecord(
        request_id=f"base_{row['query_id']}",
        query_text=row['query_text'],
        cache_hit=False,
        similarity_score=None,
        selected_route="large",
        is_mock=True,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        generated_answer=resp.text,
        reference_answer=row['expected_answer']
    )
    tracker_baseline.record_request(rec)

summary_baseline = tracker_baseline.get_summary()
print("\\n=== NO-CACHE BASELINE SUMMARY ===")
for k, v in summary_baseline.items():
    print(f"{k}: {v}")"""))

    # Exact match cache
    cells.append(nbf.v4.new_markdown_cell("""### 4. Implementing and Testing an Exact-Match Cache
An exact-match cache keys on normalized string equality (`query.strip().lower()`).
Let's see what happens when canonical questions are cached, and users ask paraphrases!"""))

    cells.append(nbf.v4.new_code_cell("""exact_cache = ExactMatchCache()
tracker_exact = CostTracker()

# Step 1: Pre-populate cache with canonical queries
canonicals = sample_queries[sample_queries['query_role'] == 'canonical']
for _, row in canonicals.iterrows():
    exact_cache.store(row['query_text'], row['expected_answer'], is_cacheable=row['is_cacheable'])

print(f"Pre-populated exact cache with {len(canonicals)} canonical questions.\\n")

# Step 2: Now simulate incoming traffic of paraphrases
for _, row in sample_queries.iterrows():
    lookup = exact_cache.lookup(row['query_text'])
    
    if lookup.hit:
        rec = RequestRecord(
            request_id=f"exact_{row['query_id']}",
            query_text=row['query_text'],
            cache_hit=True,
            similarity_score=1.0,
            selected_route="cache",
            is_mock=False,
            input_tokens=0,
            output_tokens=0,
            latency_seconds=lookup.total_lookup_latency,
            estimated_cost_usd=0.0,
            generated_answer=lookup.response
        )
    else:
        # Cache miss: generate answer
        t_start = time.perf_counter()
        resp = provider.generate(row['query_text'], model_type="large")
        lat = time.perf_counter() - t_start
        cost = tracker_exact.calculate_cost("large", resp.input_tokens, resp.output_tokens, False)
        rec = RequestRecord(
            request_id=f"exact_{row['query_id']}",
            query_text=row['query_text'],
            cache_hit=False,
            similarity_score=None,
            selected_route="large",
            is_mock=True,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_seconds=lat,
            estimated_cost_usd=cost,
            generated_answer=resp.text
        )
    tracker_exact.record_request(rec)

summary_exact = tracker_exact.get_summary()
print("=== EXACT-MATCH CACHE SUMMARY ===")
print(f"Total Requests: {summary_exact['total_requests']}")
print(f"Cache Hits: {summary_exact['cache_hits']}")
print(f"Cache Misses: {summary_exact['cache_misses']}")
print(f"Hit Rate: {summary_exact['cache_hit_rate'] * 100:.1f}%")
print(f"Total Cost: ${summary_exact['total_cost_usd']:.6f}")"""))

    # Interpretation & Interview Concepts
    cells.append(nbf.v4.new_markdown_cell("""### 5. Interpretation & Limitations of Exact-Match Caching
Notice that the exact-match cache only registers hits on verbatim matches. When a user asks:
- *"Can you tell me the capital city of France?"* instead of *"What is the capital of France?"*
The exact-match cache reports a complete **cache miss**, forwarding the duplicate request to the LLM. In typical real-world traffic, verbatim duplication is low (< 10%), while semantic duplication can reach 30%–60%.

---
### 6. Interview Concepts Checklist
Be prepared to explain these concepts in a Machine Learning Engineer interview:
1. **The Cost of Generative Inference**: Why LLM inference is orders of magnitude more expensive than embedding lookup or tabular inference (autoregressive token-by-token generation vs. forward pass).
2. **P50 vs P95 Latency**: Why mean latency hides tail latency spikes, and why caching specifically flattens tail latency.
3. **Exact vs Semantic Matching**: The failure modes of string hashing ($O(1)$ lookup with MD5/SHA256) when exposed to human linguistic variety.
4. **Cache Stampede / Thundering Herd**: What happens when multiple concurrent requests miss the cache simultaneously for the same expensive prompt."""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "01_project_setup_and_baseline.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 01_project_setup_and_baseline.ipynb")


def create_notebook_02():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 2: Semantic Similarity Experiments
## SmartCache ML: Sentence Embeddings & Cosine Similarity Calibration

### 1. Problem Overview
Exact-match caching fails when queries differ syntactically but share identical intent. To detect semantic similarity, we map query strings into continuous dense vector representations using a Sentence Transformer model (`all-MiniLM-L6-v2`).

However, semantic matching introduces a fundamental ML trade-off:
- If our similarity threshold is **too low**, we suffer **false positives** (reusing answers for questions that look similar but ask for different things).
- If our similarity threshold is **too high**, we suffer **false negatives** (missing valid paraphrases and paying unnecessary inference cost).

In this notebook, we systematically evaluate similarity thresholds to calibrate our cache.

---
### Learning Objectives
By the end of this notebook, you will be able to:
- Explain dense sentence embeddings and bi-encoder architecture.
- Compute cosine similarity using pure NumPy with $L_2$-normalized dot products.
- Evaluate similarity distributions across positive paraphrase pairs and adversarial negative pairs.
- Calibrate the similarity threshold using Precision, Recall, and F1 curves.
- Defend your threshold choice in an engineering interview."""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smartcache.config import settings
from smartcache.embeddings import encode_query, compute_cosine_similarity, compute_matrix_cosine_similarities

print("Embedding model:", settings.embedding_model_name)"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Computing Embeddings and Cosine Similarity
Let's test our embedding engine on canonical questions, true paraphrases, and adversarial distractors."""))

    cells.append(nbf.v4.new_code_cell("""q_canonical = "What is the capital of France?"
q_paraphrase = "Can you tell me the capital city of France?"
q_adversarial = "What is the largest port city in France?"
q_unrelated = "How do you calculate compound interest on an investment?"

vec_can = encode_query(q_canonical)
vec_para = encode_query(q_paraphrase)
vec_adv = encode_query(q_adversarial)
vec_unrelated = encode_query(q_unrelated)

print(f"Embedding dimension: {vec_can.shape[0]}")
print(f"Similarity (Canonical <-> Paraphrase):  {compute_cosine_similarity(vec_can, vec_para):.4f}")
print(f"Similarity (Canonical <-> Adversarial): {compute_cosine_similarity(vec_can, vec_adv):.4f}")
print(f"Similarity (Canonical <-> Unrelated):   {compute_cosine_similarity(vec_can, vec_unrelated):.4f}")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Measuring Distributions on Positive vs Negative Pairs
Let's load the benchmark dataset, pair queries within the same group (positives) and across adversarial/different groups (negatives), and compute similarity scores."""))

    cells.append(nbf.v4.new_code_cell("""df = pd.read_csv(settings.data_raw_path / "benchmark_dataset.csv")

# Collect positive pairs (queries in same group with is_semantic_match=True)
positive_pairs = []
groups = df.groupby('paraphrase_group')
for _, grp in groups:
    queries = grp[grp['is_semantic_match'] == True]['query_text'].tolist()
    if len(queries) >= 2:
        for i in range(len(queries)):
            for j in range(i + 1, min(i + 3, len(queries))):
                positive_pairs.append((queries[i], queries[j]))

# Collect negative adversarial pairs (adversarial queries vs canonicals)
negative_pairs = []
canonicals = df[df['query_role'] == 'canonical']
adversarials = df[df['query_role'] == 'adversarial_negative']

for _, adv_row in adversarials.iterrows():
    # Pair with matching canonical prefix
    base_grp = adv_row['paraphrase_group'].replace('_adv', '')
    matching_can = canonicals[canonicals['paraphrase_group'] == base_grp]
    if not matching_can.empty:
        negative_pairs.append((matching_can.iloc[0]['query_text'], adv_row['query_text']))

# Add some cross-category random negative pairs
unrelated_samples = df.sample(min(40, len(df)), random_state=42)
for i in range(0, len(unrelated_samples) - 1, 2):
    negative_pairs.append((unrelated_samples.iloc[i]['query_text'], unrelated_samples.iloc[i+1]['query_text']))

print(f"Collected {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.")"""))

    cells.append(nbf.v4.new_code_cell("""pos_scores = [compute_cosine_similarity(encode_query(p[0]), encode_query(p[1])) for p in positive_pairs]
neg_scores = [compute_cosine_similarity(encode_query(p[0]), encode_query(p[1])) for p in negative_pairs]

plt.figure(figsize=(9, 4))
plt.hist(pos_scores, bins=20, alpha=0.7, label='True Paraphrases (Positives)', color='forestgreen')
plt.hist(neg_scores, bins=20, alpha=0.7, label='Adversarial / Unrelated (Negatives)', color='firebrick')
plt.axvline(0.85, color='black', linestyle='--', label='Default Threshold (0.85)')
plt.title("Cosine Similarity Score Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Pair Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Threshold Calibration: Precision, Recall, and F1 Sweep
We evaluate candidate thresholds from 0.70 to 0.95 to find the optimal operating point."""))

    cells.append(nbf.v4.new_code_cell("""thresholds = np.linspace(0.65, 0.95, 31)
precisions = []
recalls = []
f1_scores = []

for t in thresholds:
    tp = sum(1 for s in pos_scores if s >= t)
    fn = sum(1 for s in pos_scores if s < t)
    fp = sum(1 for s in neg_scores if s >= t)
    tn = sum(1 for s in neg_scores if s < t)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

best_idx = int(np.argmax(f1_scores))
optimal_threshold = thresholds[best_idx]
print(f"Optimal Threshold by F1: {optimal_threshold:.2f}")
print(f"At {optimal_threshold:.2f} -> Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions, label='Precision (Low False Hits)', color='blue')
plt.plot(thresholds, recalls, label='Recall (High Cache Hit Rate)', color='orange')
plt.plot(thresholds, f1_scores, label='F1 Score', color='green', linewidth=2)
plt.axvline(optimal_threshold, color='red', linestyle=':', label=f'Optimal: {optimal_threshold:.2f}')
plt.xlabel("Cosine Similarity Threshold")
plt.ylabel("Score")
plt.title("Threshold Calibration Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. False Positive & False Negative Edge Cases
Let's inspect concrete failure cases:
1. **False Positive**: Two queries score above threshold but have different real-world meaning.
2. **False Negative**: Two queries have identical intent but score below threshold due to vocabulary divergence."""))

    cells.append(nbf.v4.new_code_cell("""# Inspect adversarial false positives at relaxed threshold (e.g. 0.75)
print("=== ADVERSARIAL PAIRS SCORING ABOVE 0.75 ===")
for p in negative_pairs[:10]:
    sim = compute_cosine_similarity(encode_query(p[0]), encode_query(p[1]))
    if sim >= 0.75:
        print(f"Score: {sim:.3f}\\n  Q1: {p[0]}\\n  Q2: {p[1]}\\n")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 6. Interview Concepts Checklist
1. **Bi-encoder vs Cross-encoder**: Why bi-encoders allow precomputing vectors for $O(1)$/$O(N)$ dot product search, while cross-encoders require full forward-pass concatenation ($O(N \\times \\text{model})$).
2. **Precision-Recall Trade-off in Caching**: A false cache hit serves erroneous information to the user (severe quality bug), whereas a false cache miss merely costs minor compute/latency. Therefore, caching favors high precision (e.g. threshold 0.85).
3. **L2 Normalization & Dot Product**: Why normalizing vectors to unit length transforms cosine similarity into a single hardware-accelerated matrix multiplication."""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "02_semantic_similarity_experiments.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 02_semantic_similarity_experiments.ipynb")


def create_notebook_03():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 3: Semantic Cache Evaluation
## SmartCache ML: Persistent SQLite Cache, Latency Breakdown & Isolation

### 1. Problem Overview
A production semantic cache requires more than in-memory similarity:
1. **Durability**: Entries must survive service restarts (backed by SQLite).
2. **Latency transparency**: We must isolate embedding generation time from similarity search time.
3. **Multi-tenancy / Privacy**: Queries must be partitioned by `namespace` so private user data is never cross-served.
4. **Lifecycle management**: Cache entries must respect TTL expiration and LRU eviction limits.
5. **Freshness guarantees**: Dynamic information (real-time timestamps, fluctuating prices) must be flagged non-cacheable.

---
### Learning Objectives
By the end of this notebook, you will be able to:
- Implement a persistent SQLite-backed semantic cache with vectorized NumPy similarity search.
- Dissect cache lookup latency into embedding generation vs vector search components.
- Verify multi-tenant privacy via namespace isolation.
- Demonstrate TTL expiration and LRU capacity eviction.
- Formulate safety policies regarding what should and should not be cached."""))

    cells.append(nbf.v4.new_code_cell("""import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from smartcache.config import settings
from smartcache.cache import SemanticCache, ExactMatchCache
from smartcache.cost_tracker import CostTracker

# Use temporary notebook cache db
notebook_db = settings.project_root / "data" / "notebook_cache.db"
if notebook_db.exists():
    notebook_db.unlink()

cache = SemanticCache(db_path=notebook_db, similarity_threshold=0.85, max_cache_size=100)
print(f"Initialized SemanticCache with SQLite db: {notebook_db}")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Dissecting Lookup Latency: Embedding vs Vector Search
Let's measure the exact latency breakdown between:
- Embedding model encoding (CPU tensor forward pass).
- Matrix-vector cosine search over cached items."""))

    cells.append(nbf.v4.new_code_cell("""# Populate cache with 50 diverse items
df = pd.read_csv(settings.data_raw_path / "benchmark_dataset.csv")
canonicals = df[df['query_role'] == 'canonical'].head(50)

for _, row in canonicals.iterrows():
    cache.store(row['query_text'], row['expected_answer'], is_cacheable=row['is_cacheable'])

print(f"Cached entries: {len(cache._ids)}")

# Profile 20 lookups
embed_times = []
search_times = []
total_times = []

for _, row in df.head(20).iterrows():
    res = cache.lookup(row['query_text'])
    embed_times.append(res.embedding_latency * 1000.0)
    search_times.append(res.search_latency * 1000.0)
    total_times.append(res.total_lookup_latency * 1000.0)

print(f"Mean Embedding Latency:     {np.mean(embed_times):.2f} ms")
print(f"Mean Vector Search Latency: {np.mean(search_times):.3f} ms")
print(f"Mean Total Lookup Latency:  {np.mean(total_times):.2f} ms")
print(f"Vector search represents {np.mean(search_times) / np.mean(total_times) * 100:.2f}% of total cache lookup time.")"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(6, 4))
plt.bar(['Embedding Forward Pass', 'Vector Dot-Product Search'], [np.mean(embed_times), np.mean(search_times)], color=['#1f77b4', '#2ca02c'])
plt.ylabel("Latency (milliseconds)")
plt.title("Cache Lookup Latency Breakdown")
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Namespace Isolation for Multi-Tenancy and Privacy
To prevent confidential information leaking across tenants or users, every entry is scoped to a `namespace`."""))

    cells.append(nbf.v4.new_code_cell("""# Scenario: User Alice stores private financial inquiry
cache.store(
    query="What is my current bank account balance?",
    response="Account ending in 4102 has $8,420.00.",
    namespace="tenant_user_alice",
    is_cacheable=True
)

# User Bob asks the same question in his namespace
res_bob = cache.lookup("What is my current bank account balance?", namespace="tenant_user_bob")
print(f"Bob's lookup hit status: {res_bob.hit}")
print(f"Did Bob get Alice's response? {res_bob.response}")

# Alice queries her own namespace
res_alice = cache.lookup("What is my current bank account balance?", namespace="tenant_user_alice")
print(f"Alice's lookup hit status: {res_alice.hit}")
print(f"Alice's cached response: {res_alice.response}")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. TTL Expiration & Dynamic Information Safety
Responses to queries regarding volatile information (time, stock prices, live sensor telemetry) must not be cached."""))

    cells.append(nbf.v4.new_code_cell("""# Volatile query flagged as non-cacheable
volatile_query = "What is the live stock price of Apple right now?"
stored = cache.store(volatile_query, "AAPL is $225.40", is_cacheable=False)
print(f"Attempted to store non-cacheable query -> Stored successfully? {stored}")

res_volatile = cache.lookup(volatile_query)
print(f"Lookup hit for non-cacheable query: {res_volatile.hit}")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Interview Concepts Checklist
1. **Flat NumPy Search vs Vector Databases (HNSW / FAISS)**: For caches under 50,000 vectors, in-memory NumPy matrix multiplication takes < 1 ms on CPU and avoids network hop latency. Dedicated vector DBs become necessary when cache size exceeds memory or requires sharding.
2. **Multi-Tenant Cache Partitioning**: Why logical namespacing is critical in enterprise RAG and LLM systems to avoid GDPR / HIPAA compliance breaches.
3. **Cache Eviction Policies**: Trade-offs between Least Recently Used (LRU), Least Frequently Used (LFU), and Time-To-Live (TTL) expiration."""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "03_semantic_cache_evaluation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 03_semantic_cache_evaluation.ipynb")


def create_notebook_04():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 4: Query Complexity and Cost-Aware Routing
## SmartCache ML: Routing Baseline, Scikit-Learn Classifier & Trade-offs

### 1. Problem Overview
Language models follow a steep capability vs cost curve:
- **Small models** (e.g., LLaMA-3.2-1B, Claude 3 Haiku, GPT-4o-mini) are 10x–20x cheaper and much faster, handling factual, simple, or formatting tasks cleanly.
- **Large models** (e.g., LLaMA-3.2-3B, Claude 3.5 Sonnet, GPT-4o) excel at multi-step reasoning, mathematical derivations, and complex system architectures, but are expensive and slower.

Routing queries intelligently according to **query complexity** reduces total system cost while preserving high answer quality.

---
### Learning Objectives
By the end of this notebook, you will be able to:
- Implement a rule-based routing baseline using query length and reasoning keyword heuristics.
- Train a lightweight Scikit-learn TF-IDF + Logistic Regression complexity classifier.
- Prevent data leakage by evaluating on group-partitioned train/validation/test sets.
- Analyze classifier evaluation metrics: Precision, Recall, F1, and Confusion Matrix.
- Configure probability routing thresholds to balance cost savings against risk of quality degradation."""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from smartcache.config import settings
from smartcache.router import RuleBasedRouter, ClassifierRouter

# Load group-aware dataset splits
train_df = pd.read_csv(settings.data_processed_path / "train_queries.csv")
val_df = pd.read_csv(settings.data_processed_path / "val_queries.csv")
test_df = pd.read_csv(settings.data_processed_path / "test_queries.csv")

print(f"Train samples: {len(train_df)}")
print(f"Val samples:   {len(val_df)}")
print(f"Test samples:  {len(test_df)}")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Rule-Based Routing Baseline
Let's evaluate our heuristic baseline on the test set."""))

    cells.append(nbf.v4.new_code_cell("""rule_router = RuleBasedRouter(length_threshold_words=12)

y_test_true = test_df['complexity_label'].tolist()
y_test_rule = []

for q in test_df['query_text']:
    dec = rule_router.route(q)
    # Map 'small' -> 'simple', 'large' -> 'complex'
    pred_label = "simple" if dec.selected_route == "small" else "complex"
    y_test_rule.append(pred_label)

print("=== RULE-BASED ROUTER CLASSIFICATION REPORT ===")
print(classification_report(y_test_true, y_test_rule, digits=3))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Training the Scikit-Learn TF-IDF + Logistic Regression Classifier
Now let's train our lightweight ML classifier pipeline on the training split and evaluate it on the unseen test split."""))

    cells.append(nbf.v4.new_code_cell("""ml_router = ClassifierRouter(probability_threshold=0.50)
ml_router.train(train_df['query_text'], train_df['complexity_label'])

# Evaluate on test set
y_test_ml = []
confidences = []

for q in test_df['query_text']:
    dec = ml_router.route(q)
    pred_label = "simple" if dec.selected_route == "small" else "complex"
    y_test_ml.append(pred_label)
    confidences.append(dec.confidence)

print("=== TF-IDF + LOGISTIC REGRESSION CLASSIFICATION REPORT ===")
print(classification_report(y_test_true, y_test_ml, digits=3))"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))

cm_rule = confusion_matrix(y_test_true, y_test_rule, labels=["simple", "complex"])
cm_ml = confusion_matrix(y_test_true, y_test_ml, labels=["simple", "complex"])

ConfusionMatrixDisplay(cm_rule, display_labels=["simple", "complex"]).plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title("Rule-Based Baseline")

ConfusionMatrixDisplay(cm_ml, display_labels=["simple", "complex"]).plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title("ML Classifier Router")

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Model Interpretability: Top Feature Coefficients
Let's inspect which n-grams drive the classifier toward 'simple' vs 'complex'."""))

    cells.append(nbf.v4.new_code_cell("""tfidf = ml_router.pipeline.named_steps['tfidf']
clf = ml_router.pipeline.named_steps['clf']

feature_names = np.array(tfidf.get_feature_names_out())
coefs = clf.coef_[0]

# If 'complex' is class 1 (or 0)
classes = list(clf.classes_)
complex_idx = classes.index("complex")
if complex_idx == 1:
    top_complex = feature_names[np.argsort(coefs)[-10:]]
    top_simple = feature_names[np.argsort(coefs)[:10]]
else:
    top_complex = feature_names[np.argsort(coefs)[:10]]
    top_simple = feature_names[np.argsort(coefs)[-10:]]

print("Top indicators of COMPLEX queries:", list(top_complex))
print("Top indicators of SIMPLE queries: ", list(top_simple))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Interview Concepts Checklist
1. **Model Cascading vs Ensembling**: In cascading/routing, only ONE model is queried per request based on pre-filtering, drastically cutting inference cost. In ensembling, all models execute concurrently.
2. **Threshold Tuning for Risk Asymmetry**: If a complex query is misclassified as simple, answer quality degrades. Therefore, setting a conservative routing threshold (e.g. routing to large model whenever $P(\\text{complex}) \\ge 0.35$) provides safety against hallucinations.
3. **Cold-start & Inference Latency of the Router**: The TF-IDF + LogisticRegression pipeline executes in < 0.5 ms on CPU, adding virtually zero overhead to the query pipeline."""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "04_query_complexity_and_routing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 04_query_complexity_and_routing.ipynb")


def create_notebook_05():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 5: Cost, Latency, and Quality Analysis
## SmartCache ML: Multi-Policy Benchmark & Pareto Trade-offs

### 1. Problem Overview
An optimization is only genuinely effective if its benefits (cost reduction and latency reduction) are achieved without unacceptable degradation in response quality.

In this notebook, we execute a controlled benchmark over **identical test workloads** comparing five distinct policies:
1. **Policy 1: No-Cache Baseline (Always Large Model)**: Full capability, highest cost, highest latency.
2. **Policy 2: Exact-Match Cache + Always Large**: Tests exact string cache behavior.
3. **Policy 3: Semantic Cache + Always Large**: Tests semantic cache impact in isolation.
4. **Policy 4: Always Small Model (No Cache)**: Lowest cost without caching, but lower answer quality on complex tasks.
5. **Policy 5: SmartCache ML (Semantic Cache + Cost-Aware Router)**: Full integrated system.

---
### Learning Objectives
By the end of this notebook, you will be able to:
- Design a controlled benchmark replay workload.
- Record fine-grained per-request metrics: p50/p95 latency, cost, and token overlap quality.
- Compute percentage cost savings and latency reduction against the baseline.
- Plot cost vs quality and latency vs quality Pareto frontiers.
- Defend trade-offs with empirical data in an engineering interview."""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from smartcache.config import settings
from smartcache.cache import SemanticCache, ExactMatchCache
from smartcache.router import ClassifierRouter
from smartcache.llm_provider import MockProvider
from smartcache.cost_tracker import CostTracker, RequestRecord
from smartcache.metrics import calculate_token_overlap_f1, compare_policies

print("Loaded benchmarking suite.")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Constructing the Benchmark Workload
We construct a representative replay stream containing canonical queries, paraphrases, and adversarial distractors."""))

    cells.append(nbf.v4.new_code_cell("""test_df = pd.read_csv(settings.data_processed_path / "test_queries.csv")
train_df = pd.read_csv(settings.data_processed_path / "train_queries.csv")

# Use a fixed combined workload of 50 queries with repeated paraphrases
workload = pd.concat([test_df, test_df[test_df['query_role'] == 'paraphrase']]).reset_index(drop=True)
print(f"Benchmark workload size: {len(workload)} requests.")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Executing the 5 Policies
Let's run the workload across all 5 configurations under identical conditions."""))

    cells.append(nbf.v4.new_code_cell("""def evaluate_policy(policy_name, use_semantic_cache=False, use_exact_cache=False, routing_policy="always_large"):
    tracker = CostTracker()
    provider = MockProvider()
    sem_cache = SemanticCache(db_path=settings.project_root / f"data/temp_{policy_name}.db", similarity_threshold=0.85)
    sem_cache.invalidate()
    exact_cache = ExactMatchCache()
    router = ClassifierRouter()
    
    for _, row in workload.iterrows():
        query = row['query_text']
        ref_ans = row['expected_answer']
        cacheable = row['is_cacheable']
        
        hit = False
        ans = ""
        sim_score = None
        lat = 0.0
        route_selected = "large"
        
        # 1. Cache phase
        if use_exact_cache and cacheable:
            look = exact_cache.lookup(query)
            if look.hit:
                hit = True
                ans = look.response
                lat = look.total_lookup_latency
                route_selected = "cache"
        elif use_semantic_cache and cacheable:
            look = sem_cache.lookup(query)
            sim_score = look.similarity_score
            if look.hit:
                hit = True
                ans = look.response
                lat = look.total_lookup_latency
                route_selected = "cache"
                
        # 2. Miss phase
        if not hit:
            if routing_policy == "always_large":
                route_selected = "large"
            elif routing_policy == "always_small":
                route_selected = "small"
            elif routing_policy == "router":
                dec = router.route(query)
                route_selected = dec.selected_route
                
            resp = provider.generate(query, model_type=route_selected)
            ans = resp.text
            lat = resp.latency_seconds
            
            # Store in cache
            if cacheable:
                if use_exact_cache:
                    exact_cache.store(query, ans)
                if use_semantic_cache:
                    sem_cache.store(query, ans)
                    
        cost = tracker.calculate_cost(route_selected, len(query)//4, len(ans)//4, is_cache_hit=hit)
        quality = calculate_token_overlap_f1(ans, ref_ans)
        
        rec = RequestRecord(
            request_id=f"{policy_name}_{row['query_id']}",
            query_text=query,
            cache_hit=hit,
            similarity_score=sim_score,
            selected_route=route_selected,
            is_mock=True,
            input_tokens=len(query)//4 if not hit else 0,
            output_tokens=len(ans)//4 if not hit else 0,
            latency_seconds=lat,
            estimated_cost_usd=cost,
            reference_answer=ref_ans,
            generated_answer=ans,
            quality_score=quality
        )
        tracker.record_request(rec)
        
    summary = tracker.get_summary()
    summary['mean_quality_f1'] = float(np.mean([r.quality_score for r in tracker.records]))
    return summary"""))

    cells.append(nbf.v4.new_code_cell("""res_base = evaluate_policy("base_always_large", use_semantic_cache=False, routing_policy="always_large")
res_exact = evaluate_policy("exact_cache", use_exact_cache=True, routing_policy="always_large")
res_sem = evaluate_policy("semantic_cache", use_semantic_cache=True, routing_policy="always_large")
res_small = evaluate_policy("always_small", use_semantic_cache=False, routing_policy="always_small")
res_smart = evaluate_policy("smartcache_ml", use_semantic_cache=True, routing_policy="router")

results_table = pd.DataFrame([
    {"Policy": "1. Baseline (Always Large, No Cache)", **res_base},
    {"Policy": "2. Exact Cache + Large", **res_exact},
    {"Policy": "3. Semantic Cache + Large", **res_sem},
    {"Policy": "4. Always Small (No Cache)", **res_small},
    {"Policy": "5. SmartCache ML (Semantic + Router)", **res_smart},
])

results_table[['Policy', 'cache_hit_rate', 'total_cost_usd', 'mean_latency_ms', 'p95_latency_ms', 'mean_quality_f1']]"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Visualizing Cost vs Quality Pareto Frontier"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 5))
for _, row in results_table.iterrows():
    plt.scatter(row['total_cost_usd'], row['mean_quality_f1'], s=150, label=row['Policy'])

plt.xlabel("Total Inference Cost ($ USD)")
plt.ylabel("Answer Quality (Token F1 vs Reference)")
plt.title("Cost vs Quality Trade-off Across Policies")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Interview Concepts Checklist
1. **Pareto Optimality**: Demonstrating how SmartCache ML reaches the upper-left Pareto frontier (high quality, low cost) compared to naïve policies.
2. **P95 Tail Latency Reduction**: When semantic cache hits in < 15 ms, the entire latency distribution shifts left, cutting tail p95 latency dramatically.
3. **Quality Degradation Risk**: Why evaluating token overlap or LLM-as-a-judge quality is essential before deploying cost-saving routers in production."""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "05_cost_latency_quality_analysis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 05_cost_latency_quality_analysis.ipynb")


def create_notebook_06():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 6: End-to-End System Demo
## SmartCache ML: Service Integration, Audit Logs & Deployment Guide

### 1. Problem Overview
In this final capstone notebook, we connect all modular components into the unified **SmartCacheService**:
1. Persistent Semantic Cache with SQLite.
2. Query Complexity ML Router.
3. Interchangeable LLM Provider (MockProvider / local Ollama).
4. Real-time cost, latency, and hit-rate logging.
5. Exporting benchmark artifacts for reports and dashboards.

We also demonstrate how to launch the FastAPI REST API and the Streamlit monitoring dashboard.

---
### Learning Objectives
By the end of this notebook, you will be able to:
- Orchestrate caching, routing, inference, and telemetry in a single clean Python service.
- Replay and log production traffic traces.
- Export benchmark results for stakeholder reporting.
- Run and test the FastAPI backend and Streamlit dashboard locally."""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
from smartcache.config import settings
from smartcache.service import SmartCacheService

# Initialize end-to-end service
service = SmartCacheService()
# Reset cache for a fresh demonstration
service.cache.invalidate()
service.cost_tracker.reset()

print("SmartCacheService ready!")
print("Provider:", settings.llm_provider)
print("Embedding model:", settings.embedding_model_name)"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Replaying Production Traffic Traces
Let's simulate a sequence of diverse real-world requests and observe the service decisions."""))

    cells.append(nbf.v4.new_code_cell("""traffic_queries = [
    # 1. First-time factual query (Expect: Cache Miss -> Router 'small' -> Stored in Cache)
    "What is the capital of France?",
    
    # 2. Immediate paraphrase (Expect: Cache Hit! ~0.0 cost)
    "Can you tell me the capital city of France?",
    
    # 3. Complex reasoning query (Expect: Cache Miss -> Router 'large' -> Stored in Cache)
    "Explain how QuickSort partitions elements and its worst-case algorithmic complexity.",
    
    # 4. Paraphrase of complex query (Expect: Cache Hit! ~0.0 cost)
    "How does quicksort partition work and what is its big-O time complexity?",
    
    # 5. Adversarial near-duplicate (Expect: Cache Miss -> correctly rejected from France capital cache)
    "What is the largest port city in France?",
    
    # 6. Real-time volatile query (Expect: Cache Miss -> Router 'small' -> NOT cached)
    "What is the current date and time right now?"
]

records = []
for q in traffic_queries:
    is_cacheable = "right now" not in q.lower()
    res = service.process_query(q, is_cacheable=is_cacheable)
    records.append({
        "Query": res.query[:45] + "...",
        "Cache Hit": res.cache_hit,
        "Sim Score": f"{res.similarity_score:.3f}" if res.similarity_score else "N/A",
        "Route": res.selected_route,
        "Latency (ms)": res.latency_ms,
        "Cost ($)": f"${res.estimated_cost_usd:.6f}",
        "Reason": res.routing_reason[:45] + "..."
    })

trace_df = pd.DataFrame(records)
trace_df"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Reviewing Aggregate System Metrics"""))

    cells.append(nbf.v4.new_code_cell("""summary = service.cost_tracker.get_summary()
print("=== END-TO-END SYSTEM SUMMARY ===")
print(f"Total Requests:         {summary['total_requests']}")
print(f"Cache Hits:             {summary['cache_hits']}")
print(f"Cache Misses:           {summary['cache_misses']}")
print(f"Cache Hit Rate:         {summary['cache_hit_rate'] * 100:.1f}%")
print(f"Total Estimated Cost:   ${summary['total_cost_usd']:.6f}")
print(f"Mean Latency:           {summary['mean_latency_ms']:.2f} ms")
print(f"P50 Latency:            {summary['p50_latency_ms']:.2f} ms")
print(f"P95 Latency:            {summary['p95_latency_ms']:.2f} ms")
print(f"Route Distribution:     {summary['route_distribution']}")

# Export benchmark summary
report_path = settings.project_root / "reports" / "benchmark_summary.csv"
trace_df.to_csv(report_path, index=False)
print(f"\\nSaved execution trace to {report_path}")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Running the REST API and Streamlit Dashboard
To launch the complete interactive application:

1. **FastAPI REST API**:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   Interactive OpenAPI docs available at `http://127.0.0.1:8000/docs`.

2. **Streamlit Monitoring Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   Opens browser dashboard displaying real-time metrics, cache statistics, and policy comparisons.

---
### 5. Architectural Limitations & Production Next Steps
In an engineering interview, be honest about system limitations:
1. **SQLite Concurrency**: SQLite locks the database file during writes. For high-concurrency production deployments (> 1,000 req/sec), replace SQLite with PostgreSQL (pgvector) or Redis.
2. **Flat NumPy Search vs Hierarchical Indexes**: Flat dot product is exact and fast for < 50,000 entries. For millions of embeddings, introduce an approximate nearest neighbor (ANN) index such as HNSW or ScaNN.
3. **Embedding Drift**: Changes in the underlying embedding model require re-indexing existing cache entries.
4. **Cache Invalidation on Real-world Updates**: Semantic caches require careful invalidation when authoritative source documents change."""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "06_end_to_end_system_demo.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 06_end_to_end_system_demo.ipynb")


if __name__ == "__main__":
    create_notebook_01()
    create_notebook_02()
    create_notebook_03()
    create_notebook_04()
    create_notebook_05()
    create_notebook_06()
    print("All 6 notebooks created successfully!")
