"""
Query Complexity Classifier and Model Router for SmartCache ML.
Implements heuristic baseline router and scikit-learn TF-IDF classifier router.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from smartcache.config import settings


@dataclass
class RoutingDecision:
    selected_route: str  # 'small' or 'large'
    confidence: float
    method: str  # 'rule_based' or 'classifier'
    reason: str


class RuleBasedRouter:
    """
    Heuristic rule-based baseline router.
    Routes queries based on character/word length and complex reasoning triggers.
    """

    COMPLEXITY_TRIGGERS = [
        "explain", "derive", "proof", "prove", "architecture", "algorithm",
        "step by step", "step-by-step", "trade-off", "trade off", "decompose",
        "design", "microservice", "consensus", "optimize", "optimization",
        "deep learning", "quicksort", "mergesort", "difference between",
        "detailed", "break down", "how does", "why does"
    ]

    def __init__(self, length_threshold_words: int = 15):
        self.length_threshold = length_threshold_words

    def route(self, query: str) -> RoutingDecision:
        normalized = query.strip().lower()
        words = normalized.split()

        # Rule 1: High word count indicates complex reasoning or problem description
        if len(words) > self.length_threshold:
            return RoutingDecision(
                selected_route="large",
                confidence=0.75,
                method="rule_based",
                reason=f"Query word count ({len(words)}) exceeds threshold ({self.length_threshold})"
            )

        # Rule 2: Keyword match on technical/reasoning terms
        matched_triggers = [kw for kw in self.COMPLEXITY_TRIGGERS if kw in normalized]
        if matched_triggers:
            return RoutingDecision(
                selected_route="large",
                confidence=0.70,
                method="rule_based",
                reason=f"Matched complex triggers: {matched_triggers[:2]}"
            )

        # Default fallback to small model
        return RoutingDecision(
            selected_route="small",
            confidence=0.65,
            method="rule_based",
            reason="Short query with no complex reasoning keywords detected"
        )


class ClassifierRouter:
    """
    Machine Learning Router powered by a lightweight scikit-learn TF-IDF + LogisticRegression pipeline.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        probability_threshold: float = 0.50
    ):
        self.model_path = model_path or (settings.project_root / "reports" / "router_model.joblib")
        self.threshold = probability_threshold
        self.pipeline: Optional[Pipeline] = None
        self._load_if_exists()

    def _load_if_exists(self):
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception:
                self.pipeline = None

    def train(self, X_train, y_train):
        """
        Train TF-IDF + LogisticRegression pipeline.
        X_train: list/array of query texts
        y_train: labels ('simple' or 'complex')
        """
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=200, random_state=42))
        ])
        self.pipeline.fit(X_train, y_train)

    def save(self, destination: Optional[Path] = None):
        """Save trained pipeline artifact."""
        path = destination or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def route(self, query: str) -> RoutingDecision:
        if self.pipeline is None:
            # Fallback to rule-based if classifier model artifact is not yet trained/saved
            fallback = RuleBasedRouter().route(query)
            return RoutingDecision(
                selected_route=fallback.selected_route,
                confidence=fallback.confidence,
                method="fallback_rule_based",
                reason="Classifier pipeline not fitted; using rule-based fallback"
            )

        # Predict probability of 'complex'
        classes = list(self.pipeline.classes_)
        proba = self.pipeline.predict_proba([query])[0]
        
        if "complex" in classes:
            complex_idx = classes.index("complex")
            prob_complex = float(proba[complex_idx])
        else:
            prob_complex = 0.0

        if prob_complex >= self.threshold:
            return RoutingDecision(
                selected_route="large",
                confidence=round(prob_complex, 4),
                method="classifier",
                reason=f"Classifier predicted complex with probability {prob_complex:.3f} >= threshold {self.threshold:.2f}"
            )
        else:
            return RoutingDecision(
                selected_route="small",
                confidence=round(1.0 - prob_complex, 4),
                method="classifier",
                reason=f"Classifier predicted simple (p_complex={prob_complex:.3f} < threshold {self.threshold:.2f})"
            )
