import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class GatekeeperLR:
    def __init__(self, max_features=50000, ngram_range=(1, 2), max_iter=1000, C=1.0, random_state=42):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ('clf', LogisticRegression(random_state=self.random_state, max_iter=max_iter, C=C))
        ])

    def train(self, X_train, y_train):
        print("Training GatekeeperLR model...")
        self.pipeline.fit(X_train, y_train)
        print("GatekeeperLR trained.")

    def predict_with_threshold(self, X, threshold=0.70):
        """
        Returns predictions based on threshold.
        If P(safe) >= threshold, label=1 (safe), else label=0 (not-safe).
        Returns:
            labels: array of 0 or 1
            probs: probability of the 'safe' class (class 1)
            passed_mask: boolean array where True means the sample passed the threshold (P(safe) >= threshold)
        """
        # predict_proba returns [P(not_safe), P(safe)] assuming classes are [0, 1]
        probs_all = self.pipeline.predict_proba(X)
        # Assuming class 1 is safe
        safe_idx = list(self.pipeline.classes_).index(1) 
        probs = probs_all[:, safe_idx]
        
        passed_mask = probs >= threshold
        labels = np.where(passed_mask, 1, 0)
        
        return labels, probs, passed_mask
        
    def predict(self, X):
         return self.pipeline.predict(X)
         
    def predict_proba(self, X):
         return self.pipeline.predict_proba(X)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"GatekeeperLR saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance
