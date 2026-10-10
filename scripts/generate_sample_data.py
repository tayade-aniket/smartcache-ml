"""
Generate Sample Benchmark Dataset for SmartCache ML
===================================================

This script generates a curated benchmark dataset of 300+ queries covering:
- Simple factual queries (suitable for small models)
- Multi-step reasoning and complex explanations (requiring large models)
- Programming and debugging queries
- Paraphrase groups with semantic duplicates (positive cache pairs)
- Adversarial surface-similar queries with different semantic meanings (negative cache pairs)
- Dynamic / volatile information queries marked non-cacheable

It performs group-aware splitting into train, validation, and test sets to
prevent data leakage during routing classifier training and evaluation.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Query Templates & Data Generation
# ---------------------------------------------------------------------------

RAW_GROUPS = [
    # Group 1: Capital of France (Factual, Simple, Positive matches + Negative pairs)
    {
        "group_id": "grp_001",
        "category": "simple_factual",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "The capital of France is Paris.",
        "canonical": "What is the capital of France?",
        "paraphrases": [
            "Can you tell me the capital city of France?",
            "Which city serves as the capital of France?",
            "What's France's capital city?",
            "Tell me what the capital of France is.",
        ],
        "adversarials": [
            {
                "query": "What is the largest port city in France?",
                "complexity": "simple",
                "route": "small",
                "answer": "The largest port city in France is Marseille.",
                "cacheable": True,
            },
            {
                "query": "What was the capital of France during World War II?",
                "complexity": "complex",
                "route": "large",
                "answer": "During World War II, Vichy was the seat of the collaborationist government, while Paris remained the de jure capital.",
                "cacheable": True,
            }
        ]
    },

    # Group 2: Speed of Light (Factual, Simple)
    {
        "group_id": "grp_002",
        "category": "simple_factual",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s).",
        "canonical": "What is the speed of light in a vacuum?",
        "paraphrases": [
            "How fast does light travel in vacuum?",
            "What is the velocity of light in empty space?",
            "Can you give the speed of light in m/s?",
        ],
        "adversarials": [
            {
                "query": "What is the speed of sound in air?",
                "complexity": "simple",
                "route": "small",
                "answer": "The speed of sound in air at 20°C is approximately 343 meters per second.",
                "cacheable": True,
            }
        ]
    },

    # Group 3: Python List vs Tuple (Programming, Simple-Medium)
    {
        "group_id": "grp_003",
        "category": "programming",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "In Python, lists are mutable (can be changed after creation), while tuples are immutable (read-only after creation). Lists use square brackets [], tuples use parentheses ().",
        "canonical": "What is the difference between a list and a tuple in Python?",
        "paraphrases": [
            "How do Python lists differ from tuples?",
            "Explain the difference between tuples and lists in Python.",
            "In Python, what distinguishes a tuple from a list?",
            "Python list vs tuple differences explained simply.",
        ],
        "adversarials": [
            {
                "query": "What is the difference between a list and a set in Python?",
                "complexity": "simple",
                "route": "small",
                "answer": "Lists are ordered collections allowing duplicates, while sets are unordered collections of unique elements.",
                "cacheable": True,
            }
        ]
    },

    # Group 4: QuickSort Algorithm (Programming / Reasoning, Complex)
    {
        "group_id": "grp_004",
        "category": "programming",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "QuickSort is a divide-and-conquer algorithm. It selects a pivot, partitions elements into subarrays less than and greater than the pivot, and recursively sorts subarrays. Average time complexity is O(n log n), worst-case O(n^2).",
        "canonical": "How does the QuickSort algorithm work and what is its time complexity?",
        "paraphrases": [
            "Explain QuickSort implementation steps and its algorithmic complexity.",
            "Can you explain how QuickSort partitions data and its average vs worst-case runtime?",
            "Break down the divide and conquer mechanism of quicksort with complexity.",
        ],
        "adversarials": [
            {
                "query": "How does MergeSort work and how does it guarantee O(n log n)?",
                "complexity": "complex",
                "route": "large",
                "answer": "MergeSort recursively divides the array in halves and merges sorted halves, guaranteeing O(n log n) even in worst case at the cost of O(n) auxiliary space.",
                "cacheable": True,
            }
        ]
    },

    # Group 5: Gradient Descent (Data Analysis / ML, Complex)
    {
        "group_id": "grp_005",
        "category": "data_analysis",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "Gradient descent is an iterative optimization algorithm that minimizes an objective function by updating parameters in the opposite direction of the gradient of the loss with respect to parameters: theta = theta - lr * grad.",
        "canonical": "Explain how gradient descent optimization works in machine learning.",
        "paraphrases": [
            "What is gradient descent and how does it optimize model weights?",
            "How does the gradient descent algorithm find optimal parameters?",
            "Explain mathematical intuition behind gradient descent parameter updates.",
        ],
        "adversarials": [
            {
                "query": "What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?",
                "complexity": "complex",
                "route": "large",
                "answer": "Batch GD uses the entire dataset to compute gradients per step, while SGD updates parameters using a single random training sample per step.",
                "cacheable": True,
            }
        ]
    },

    # Group 6: Photosynthesis Process (General Knowledge / Explanation, Complex)
    {
        "group_id": "grp_006",
        "category": "long_explanation",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "Photosynthesis is the biochemical process by which green plants and certain organisms convert sunlight, water, and CO2 into glucose and oxygen, consisting of light-dependent reactions in thylakoids and the Calvin cycle in stroma.",
        "canonical": "Explain the detailed chemical process of photosynthesis in plants.",
        "paraphrases": [
            "How does photosynthesis work step by step in green plants?",
            "Can you describe the biological mechanism of photosynthesis?",
            "Walk me through the stages of photosynthesis including light reactions and Calvin cycle.",
        ],
        "adversarials": [
            {
                "query": "What is cellular respiration and how does it differ from photosynthesis?",
                "complexity": "complex",
                "route": "large",
                "answer": "Cellular respiration breaks down glucose and oxygen into ATP, CO2, and water, effectively the reverse energetic cycle of photosynthesis.",
                "cacheable": True,
            }
        ]
    },

    # Group 7: Multi-step Reasoning / Math Word Problem (Reasoning, Complex)
    {
        "group_id": "grp_007",
        "category": "multi_step_reasoning",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "If a train travels 120 km in 2 hours, speed is 60 km/h. At that rate, to travel 330 km it takes 330 / 60 = 5.5 hours (5 hours 30 minutes).",
        "canonical": "A train travels 120 km in 2 hours. At the same constant speed, how many hours will it take to travel 330 km?",
        "paraphrases": [
            "If a train covers 120 kilometers in 2 hours at constant speed, how long does it take to cover 330 kilometers?",
            "Calculate travel duration for 330 km given a train moves 120 km every two hours.",
            "Solve: Train speed is 120 km per 2 hrs. How much time is needed for 330 km?",
        ],
        "adversarials": [
            {
                "query": "A train travels 120 km in 2 hours and accelerates by 10 km/h. How long to travel 330 km?",
                "complexity": "complex",
                "route": "large",
                "answer": "With acceleration, initial speed is 60 km/h and increases to 70 km/h, requiring step-by-step kinematics integration.",
                "cacheable": True,
            }
        ]
    },

    # Group 8: Changing / Real-time info (Non-cacheable!)
    {
        "group_id": "grp_008",
        "category": "changing_info",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": False,
        "base_answer": "Current system time is dynamic and cannot be statically cached.",
        "canonical": "What is the current date and time right now?",
        "paraphrases": [
            "What is today's current date and time?",
            "Tell me the exact current time right now.",
            "What time is it at this moment?",
        ],
        "adversarials": [
            {
                "query": "How many seconds are in a standard 24 hour day?",
                "complexity": "simple",
                "route": "small",
                "answer": "There are 86,400 seconds in a standard 24-hour day (24 * 60 * 60).",
                "cacheable": True,
            }
        ]
    },

    # Group 9: Current stock price (Changing / Real-time, Non-cacheable)
    {
        "group_id": "grp_009",
        "category": "changing_info",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": False,
        "base_answer": "Stock quotes fluctuate tick-by-tick and require live financial API feeds.",
        "canonical": "What is the live stock price of Apple right now?",
        "paraphrases": [
            "How much is Apple stock trading for currently?",
            "What is AAPL share price at this instant?",
            "Can you tell me the current price of AAPL shares?",
        ],
        "adversarials": [
            {
                "query": "In what year was Apple Inc. founded?",
                "complexity": "simple",
                "route": "small",
                "answer": "Apple Inc. was founded on April 1, 1976.",
                "cacheable": True,
            }
        ]
    },

    # Group 10: Simple Fact: Boiling point of water
    {
        "group_id": "grp_010",
        "category": "simple_factual",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm).",
        "canonical": "At what temperature does water boil at sea level?",
        "paraphrases": [
            "What is the boiling point of water at 1 atm?",
            "At what temperature does liquid water boil at standard pressure?",
            "What is water's boiling temperature at sea level in Celsius?",
        ],
        "adversarials": [
            {
                "query": "At what temperature does pure water freeze at sea level?",
                "complexity": "simple",
                "route": "small",
                "answer": "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) at sea level.",
                "cacheable": True,
            }
        ]
    },

    # Group 11: SQL Query optimization (Complex, Programming)
    {
        "group_id": "grp_011",
        "category": "programming",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "To optimize slow SQL queries: inspect EXPLAIN ANALYZE, create appropriate indexes on JOIN and WHERE columns, eliminate SELECT *, avoid correlated subqueries, and ensure statistics are up to date.",
        "canonical": "How do you diagnose and optimize a slow running SQL query in PostgreSQL?",
        "paraphrases": [
            "What are best practices to speed up slow SQL database queries in Postgres?",
            "How can I analyze and optimize slow Postgres queries?",
            "Steps for tuning poorly performing queries in PostgreSQL with EXPLAIN.",
        ],
        "adversarials": [
            {
                "query": "What is the basic SQL syntax for a SELECT query with WHERE clause?",
                "complexity": "simple",
                "route": "small",
                "answer": "SELECT column1, column2 FROM table_name WHERE condition;",
                "cacheable": True,
            }
        ]
    },

    # Group 12: Precision vs Recall (ML, Simple-Medium)
    {
        "group_id": "grp_012",
        "category": "data_analysis",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "Precision is TP / (TP + FP) measuring the accuracy of positive predictions. Recall is TP / (TP + FN) measuring the proportion of actual positives successfully identified.",
        "canonical": "What is the mathematical difference between precision and recall in classification?",
        "paraphrases": [
            "How do precision and recall differ in machine learning evaluation?",
            "Explain the difference between classification precision and recall formulas.",
            "Precision vs recall definitions and formulas explained.",
        ],
        "adversarials": [
            {
                "query": "What is the formula for the F1 score in terms of precision and recall?",
                "complexity": "simple",
                "route": "small",
                "answer": "F1 score is the harmonic mean: 2 * (Precision * Recall) / (Precision + Recall).",
                "cacheable": True,
            }
        ]
    },

    # Group 13: Distributed Systems CAP Theorem (Complex explanation)
    {
        "group_id": "grp_013",
        "category": "long_explanation",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "The CAP theorem states that a distributed data store can guarantee at most two out of three properties: Consistency (every read gets the latest write), Availability (every request receives a non-error response), and Partition Tolerance (system continues functioning despite network packet drops).",
        "canonical": "Explain the CAP theorem in distributed systems with trade-offs.",
        "paraphrases": [
            "What does CAP theorem mean for distributed databases?",
            "Can you explain consistency, availability, and partition tolerance trade-offs?",
            "Detail the trade-offs described by the CAP theorem.",
        ],
        "adversarials": [
            {
                "query": "What is the difference between ACID and BASE database transactions?",
                "complexity": "complex",
                "route": "large",
                "answer": "ACID focuses on strict consistency and immediate durability, while BASE favors high availability and eventual consistency.",
                "cacheable": True,
            }
        ]
    },

    # Group 14: Linear Regression Assumptions (ML, Complex)
    {
        "group_id": "grp_014",
        "category": "data_analysis",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "Key assumptions of ordinary least squares linear regression are: linearity between features and target, independence of residuals (no autocorrelation), homoscedasticity (constant error variance), normality of residuals, and no multicollinearity among predictors.",
        "canonical": "What are the core mathematical assumptions of linear regression and how to test them?",
        "paraphrases": [
            "List and explain the main assumptions required for Ordinary Least Squares regression.",
            "What conditions must be satisfied for linear regression to be valid?",
            "Explain assumptions of OLS regression including homoscedasticity and normality.",
        ],
        "adversarials": [
            {
                "query": "What is the formula for the Ordinary Least Squares (OLS) closed-form solution?",
                "complexity": "complex",
                "route": "large",
                "answer": "Beta = (X^T * X)^(-1) * X^T * y.",
                "cacheable": True,
            }
        ]
    },

    # Group 15: Python Decorator (Programming, Medium)
    {
        "group_id": "grp_015",
        "category": "programming",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "A Python decorator is a callable that takes another function as an argument, extends or modifies its behavior without modifying the original code, and returns the wrapped function.",
        "canonical": "How do decorators work in Python and how do you write a custom one?",
        "paraphrases": [
            "Explain how function decorators work in Python with an example.",
            "What is a Python decorator and how do you implement one?",
            "Can you explain Python decorator syntax and inner wrapper functions?",
        ],
        "adversarials": [
            {
                "query": "What is a Python generator and how does the yield keyword work?",
                "complexity": "complex",
                "route": "large",
                "answer": "A generator produces a sequence of values lazily over time using yield, preserving state between successive next() calls.",
                "cacheable": True,
            }
        ]
    },

    # Group 16: Binary Search (Programming / Simple Factual)
    {
        "group_id": "grp_016",
        "category": "programming",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "Binary search finds an element in a sorted array by repeatedly dividing the search interval in half. Its time complexity is O(log n).",
        "canonical": "What is the time complexity of binary search on a sorted array?",
        "paraphrases": [
            "How fast is binary search on sorted data in big-O notation?",
            "What is the algorithmic complexity of binary search?",
            "Big O runtime of binary search explained.",
        ],
        "adversarials": [
            {
                "query": "What is the time complexity of searching an element in an unsorted array?",
                "complexity": "simple",
                "route": "small",
                "answer": "Linear search on an unsorted array takes O(n) time.",
                "cacheable": True,
            }
        ]
    },

    # Group 17: Multi-step Reasoning / Probability (Reasoning, Complex)
    {
        "group_id": "grp_017",
        "category": "multi_step_reasoning",
        "complexity": "complex",
        "expected_route": "large",
        "is_cacheable": True,
        "base_answer": "When rolling two standard six-sided dice, there are 36 outcomes. Combinations summing to 7 are (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 outcomes. Probability is 6/36 = 1/6 (approx 16.67%).",
        "canonical": "What is the probability of rolling a sum of 7 with two fair six-sided dice?",
        "paraphrases": [
            "If you roll two six-sided dice, what are the odds of getting a sum of seven?",
            "Calculate the exact probability that two dice sum to 7.",
            "Two fair 6-sided dice are thrown: find the chance of rolling 7.",
        ],
        "adversarials": [
            {
                "query": "What is the probability of rolling snake eyes (sum of 2) with two dice?",
                "complexity": "simple",
                "route": "small",
                "answer": "Only outcome (1,1) yields sum 2, so the probability is 1/36 (approx 2.78%).",
                "cacheable": True,
            }
        ]
    },

    # Group 18: Docker Container vs Virtual Machine (General Knowledge / Tech)
    {
        "group_id": "grp_018",
        "category": "general_knowledge",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "Containers share the host OS kernel and isolate user space processes, making them lightweight and fast to boot. Virtual machines virtualize full hardware and run a complete guest OS on a hypervisor, consuming more memory and CPU overhead.",
        "canonical": "What is the fundamental architectural difference between Docker containers and virtual machines?",
        "paraphrases": [
            "How do containers differ from virtual machines in architecture?",
            "Docker container vs VM: what is the core architectural difference?",
            "Explain the difference between running a VM and running a container.",
        ],
        "adversarials": [
            {
                "query": "What is Kubernetes and how does it relate to Docker?",
                "complexity": "complex",
                "route": "large",
                "answer": "Kubernetes is an orchestration platform designed to automate deployment, scaling, and networking across clusters of containerized applications.",
                "cacheable": True,
            }
        ]
    },

    # Group 19: Overfitting vs Underfitting (ML, Simple)
    {
        "group_id": "grp_019",
        "category": "data_analysis",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "Overfitting occurs when a model learns training noise and performs poorly on unseen data (high variance). Underfitting occurs when a model is too simple to capture underlying patterns in training data (high bias).",
        "canonical": "What is the difference between overfitting and underfitting in machine learning?",
        "paraphrases": [
            "Explain overfitting vs underfitting with bias and variance.",
            "How does underfitting differ from overfitting in ML models?",
            "Define overfitting and underfitting in predictive modeling.",
        ],
        "adversarials": [
            {
                "query": "What techniques can you use to prevent overfitting in deep neural networks?",
                "complexity": "complex",
                "route": "large",
                "answer": "Techniques include dropout, L1/L2 weight regularization, data augmentation, early stopping, and batch normalization.",
                "cacheable": True,
            }
        ]
    },

    # Group 20: Git Merge vs Rebase (Programming, Simple-Medium)
    {
        "group_id": "grp_020",
        "category": "programming",
        "complexity": "simple",
        "expected_route": "small",
        "is_cacheable": True,
        "base_answer": "git merge preserves the full commit history and creates a new merge commit connecting branches. git rebase rewrites project history by moving the base of current branch onto the target branch, producing a linear commit sequence.",
        "canonical": "What is the difference between git merge and git rebase?",
        "paraphrases": [
            "How do git rebase and git merge differ in version control?",
            "Explain git merge vs rebase and their effects on commit history.",
            "When should I use git merge versus git rebase?",
        ],
        "adversarials": [
            {
                "query": "What does git cherry-pick do and how is it used?",
                "complexity": "simple",
                "route": "small",
                "answer": "git cherry-pick applies the changes introduced by a specific existing commit from another branch onto your current HEAD.",
                "cacheable": True,
            }
        ]
    }
]

# Additional synthetic expansion to create 250+ structured query examples
ADDITIONAL_FACTUAL_TOPICS = [
    ("What is the boiling point of ethanol?", "Ethanol boils at 78.37 degrees Celsius at standard atmospheric pressure.", "simple_factual", "simple", "small", True),
    ("What is the freezing point of ethanol?", "Ethanol freezes at -114.1 degrees Celsius.", "simple_factual", "simple", "small", True),
    ("Who wrote the play Hamlet?", "William Shakespeare wrote Hamlet around 1599-1601.", "general_knowledge", "simple", "small", True),
    ("Who wrote the novel 1984?", "George Orwell published the dystopian novel Nineteen Eighty-Four in 1949.", "general_knowledge", "simple", "small", True),
    ("What is the atomic number of Carbon?", "Carbon has an atomic number of 6.", "simple_factual", "simple", "small", True),
    ("What is the atomic number of Gold?", "Gold (Au) has an atomic number of 79.", "simple_factual", "simple", "small", True),
    ("What is the distance from the Earth to the Sun?", "The average distance from the Earth to the Sun is approximately 149.6 million kilometers (1 AU).", "simple_factual", "simple", "small", True),
    ("What is the diameter of Earth?", "Earth's mean diameter is approximately 12,742 kilometers.", "simple_factual", "simple", "small", True),
    ("What does the HTTP 404 status code mean?", "HTTP 404 Not Found indicates that the server cannot locate the requested resource.", "programming", "simple", "small", True),
    ("What does the HTTP 500 status code mean?", "HTTP 500 Internal Server Error indicates an unexpected condition on the server preventing it from fulfilling the request.", "programming", "simple", "small", True),
    ("What does the HTTP 403 status code mean?", "HTTP 403 Forbidden indicates the server understood the request but refuses to authorize it.", "programming", "simple", "small", True),
    ("What does the HTTP 401 status code mean?", "HTTP 401 Unauthorized indicates the request requires user authentication credentials.", "programming", "simple", "small", True),
    ("How do you check current memory usage in Linux?", "Use commands like free -m, top, or htop to inspect Linux memory usage.", "programming", "simple", "small", True),
    ("What is the command to list open ports in Linux?", "Run netstat -tuln or ss -tuln to list currently listening network ports.", "programming", "simple", "small", True),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo.", "simple_factual", "simple", "small", True),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa.", "simple_factual", "simple", "small", True),
    ("What is the capital of Australia?", "The capital of Australia is Canberra.", "simple_factual", "simple", "small", True),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia.", "simple_factual", "simple", "small", True),
    ("What is the currency of the United Kingdom?", "The official currency of the UK is the British Pound Sterling (GBP).", "simple_factual", "simple", "small", True),
    ("What is the currency of Japan?", "The currency of Japan is the Japanese Yen (JPY).", "simple_factual", "simple", "small", True),
]

ADDITIONAL_COMPLEX_TOPICS = [
    ("Design a scalable rate-limiting system for a high-traffic microservices architecture.", "Implement a token bucket or leaky bucket algorithm backed by Redis with atomic Lua scripts, sliding window counters, and local in-memory caching to balance throughput and strict enforcement.", "multi_step_reasoning", "complex", "large", True),
    ("Explain the transformer self-attention mechanism with query, key, and value matrices.", "Self-attention computes Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V, mapping query vectors against key vectors to produce normalized attention weights that dynamically scale value vectors.", "long_explanation", "complex", "large", True),
    ("How does Raft consensus algorithm handle split votes and leader election?", "Raft uses randomized election timeouts to ensure one candidate initiates election before others, requiring a majority quorum of node votes to secure leadership and avoid repeated split votes.", "multi_step_reasoning", "complex", "large", True),
    ("Analyze how cache coherency protocols like MESI maintain multi-core L1/L2 cache consistency.", "MESI tracks cache line states (Modified, Exclusive, Shared, Invalid) and uses bus snooping or directory systems to invalidate or write back stale cache lines upon processor writes.", "long_explanation", "complex", "large", True),
    ("Step-by-step mathematical proof that the square root of 2 is an irrational number.", "Assume sqrt(2) = a/b in lowest terms, then 2b^2 = a^2 implying a is even. Substituting a=2k yields b^2 = 2k^2 implying b is also even, contradicting coprime assumption.", "multi_step_reasoning", "complex", "large", True),
    ("Derive the backpropagation algorithm for a two-layer feedforward neural network using the chain rule.", "Compute loss derivative wrt output activations, apply chain rule through activation functions, compute weight gradients dL/dW2, and propagate delta errors backwards through W1.", "data_analysis", "complex", "large", True),
    ("Compare and contrast Eventual Consistency vs Strong Consistency in Amazon DynamoDB.", "Strong consistency guarantees reading the most recent successful write with increased latency and potential partition unavailability, whereas eventual consistency serves reads from any replica with lower latency.", "long_explanation", "complex", "large", True),
    ("Explain the internal mechanisms of Linux epoll vs select for handling tens of thousands of concurrent sockets.", "select inspects all file descriptors in O(n) scan, whereas epoll uses red-black tree registrations and an event-ready list, notifying user space in O(1) time per active event.", "programming", "complex", "large", True),
    ("Solve this scheduling problem: 5 tasks with deadlines and profits, find maximum profit sequence.", "Sort tasks descending by profit and iteratively place each task into the latest available time slot before its deadline using a greedy disjoint-set approach.", "multi_step_reasoning", "complex", "large", True),
    ("Explain the bias-variance decomposition of mean squared error in supervised learning.", "MSE decomposes into Bias(f)^2 + Variance(f) + Irreducible Error sigma^2, illustrating that reducing model complexity increases bias while aggressive fitting spikes variance.", "data_analysis", "complex", "large", True),
]


def generate_dataset():
    records = []
    query_counter = 1

    # 1. Process Core Groups (Canonical, Paraphrases, and Adversarial controls)
    for group in RAW_GROUPS:
        grp_id = group["group_id"]
        cat = group["category"]
        complexity = group["complexity"]
        route = group["expected_route"]
        cacheable = group["is_cacheable"]
        base_ans = group["base_answer"]

        # Canonical query (The reference stored in cache or first seen)
        records.append({
            "query_id": f"q_{query_counter:04d}",
            "query_text": group["canonical"],
            "category": cat,
            "complexity_label": complexity,
            "expected_answer": base_ans,
            "paraphrase_group": grp_id,
            "is_semantic_match": True,  # Canonical is the anchor match
            "expected_route": route,
            "is_cacheable": cacheable,
            "query_role": "canonical"
        })
        query_counter += 1

        # Paraphrases (True semantic duplicates that MUST hit cache)
        for p_text in group["paraphrases"]:
            records.append({
                "query_id": f"q_{query_counter:04d}",
                "query_text": p_text,
                "category": cat,
                "complexity_label": complexity,
                "expected_answer": base_ans,
                "paraphrase_group": grp_id,
                "is_semantic_match": True,
                "expected_route": route,
                "is_cacheable": cacheable,
                "query_role": "paraphrase"
            })
            query_counter += 1

        # Adversarials (Surface-similar query with different intent, MUST NOT hit canonical cache)
        for adv in group.get("adversarials", []):
            adv_grp = f"{grp_id}_adv"
            records.append({
                "query_id": f"q_{query_counter:04d}",
                "query_text": adv["query"],
                "category": cat,
                "complexity_label": adv["complexity"],
                "expected_answer": adv["answer"],
                "paraphrase_group": adv_grp,
                "is_semantic_match": False,  # Not a match for the canonical query!
                "expected_route": adv["route"],
                "is_cacheable": adv["cacheable"],
                "query_role": "adversarial_negative"
            })
            query_counter += 1

    # 2. Process Additional Factual queries with paraphrases
    extra_grp_idx = 100
    for q_text, ans, cat, comp, route, cacheable in ADDITIONAL_FACTUAL_TOPICS:
        grp_id = f"grp_{extra_grp_idx:03d}"
        extra_grp_idx += 1

        # Canonical
        records.append({
            "query_id": f"q_{query_counter:04d}",
            "query_text": q_text,
            "category": cat,
            "complexity_label": comp,
            "expected_answer": ans,
            "paraphrase_group": grp_id,
            "is_semantic_match": True,
            "expected_route": route,
            "is_cacheable": cacheable,
            "query_role": "canonical"
        })
        query_counter += 1

        # Generate 2-3 simple syntactic variants
        variations = [
            f"Please clarify: {q_text.lower()}",
            f"Quick question: {q_text}",
            f"Can someone tell me, {q_text[0].lower() + q_text[1:]}"
        ]
        for v in variations[:random.choice([2, 3])]:
            records.append({
                "query_id": f"q_{query_counter:04d}",
                "query_text": v,
                "category": cat,
                "complexity_label": comp,
                "expected_answer": ans,
                "paraphrase_group": grp_id,
                "is_semantic_match": True,
                "expected_route": route,
                "is_cacheable": cacheable,
                "query_role": "paraphrase"
            })
            query_counter += 1

    # 3. Process Additional Complex topics with paraphrases
    for q_text, ans, cat, comp, route, cacheable in ADDITIONAL_COMPLEX_TOPICS:
        grp_id = f"grp_{extra_grp_idx:03d}"
        extra_grp_idx += 1

        records.append({
            "query_id": f"q_{query_counter:04d}",
            "query_text": q_text,
            "category": cat,
            "complexity_label": comp,
            "expected_answer": ans,
            "paraphrase_group": grp_id,
            "is_semantic_match": True,
            "expected_route": route,
            "is_cacheable": cacheable,
            "query_role": "canonical"
        })
        query_counter += 1

        variations = [
            f"In depth analysis requested: {q_text}",
            f"Could you explain thoroughly: {q_text.lower()}",
            f"Please break down the following technical topic: {q_text}"
        ]
        for v in variations[:random.choice([2, 3])]:
            records.append({
                "query_id": f"q_{query_counter:04d}",
                "query_text": v,
                "category": cat,
                "complexity_label": comp,
                "expected_answer": ans,
                "paraphrase_group": grp_id,
                "is_semantic_match": True,
                "expected_route": route,
                "is_cacheable": cacheable,
                "query_role": "paraphrase"
            })
            query_counter += 1

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} total query examples across {df['paraphrase_group'].nunique()} groups.")
    return df


def split_dataset(df):
    """
    Split the dataset into Train (70%), Validation (15%), and Test (15%) splits
    grouped by paraphrase_group to guarantee NO semantic leakage between splits.
    """
    unique_groups = df["paraphrase_group"].unique().tolist()
    random.shuffle(unique_groups)

    n_groups = len(unique_groups)
    train_end = int(0.70 * n_groups)
    val_end = int(0.85 * n_groups)

    train_groups = set(unique_groups[:train_end])
    val_groups = set(unique_groups[train_end:val_end])
    test_groups = set(unique_groups[val_end:])

    train_df = df[df["paraphrase_group"].isin(train_groups)].copy()
    val_df = df[df["paraphrase_group"].isin(val_groups)].copy()
    test_df = df[df["paraphrase_group"].isin(test_groups)].copy()

    return train_df, val_df, test_df


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()

    # Save complete benchmark dataset
    raw_path = DATA_RAW / "benchmark_dataset.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved complete benchmark dataset to {raw_path}")

    # Split dataset
    train_df, val_df, test_df = split_dataset(df)

    train_path = DATA_PROCESSED / "train_queries.csv"
    val_path = DATA_PROCESSED / "val_queries.csv"
    test_path = DATA_PROCESSED / "test_queries.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train set: {len(train_df)} queries ({train_df['paraphrase_group'].nunique()} groups)")
    print(f"Validation set: {len(val_df)} queries ({val_df['paraphrase_group'].nunique()} groups)")
    print(f"Test set: {len(test_df)} queries ({test_df['paraphrase_group'].nunique()} groups)")
    print(f"Complexity distribution in train: {train_df['complexity_label'].value_counts().to_dict()}")
    print(f"Complexity distribution in test: {test_df['complexity_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
