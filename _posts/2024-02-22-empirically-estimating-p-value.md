---
title: Empirically Estimating p-value
date: 2024-02-21 08:00:00 +0100
categories: [Data Science, Model Evaluation & Selection]
tags: [p-value, significance, criterion]
math: true
---

```python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)


from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()

# dummy_clf.fit(X, y)

from sklearn.model_selection import permutation_test_score

score, permutation_scores, pvalue = permutation_test_score(dummy_clf, X, y, random_state=0)

print(f"P-value: {pvalue:.3f}")

print(f"Original Score: {score:.3f}")

print(f"Permutation Scores: {permutation_scores.mean():.3f} +/- {permutation_scores.std():.3f}")

```

    P-value: 1.000
    Original Score: 0.333
    Permutation Scores: 0.333 +/- 0.000



```python

from sklearn.ensemble import HistGradientBoostingClassifier

hist_clf = HistGradientBoostingClassifier().fit(X, y)

score, permutation_scores, pvalue = permutation_test_score(hist_clf, X, y, random_state=0)



print(f"P-value: {pvalue:.3f}")

print(f"Original Score: {score:.3f}")

print(f"Permutation Scores: {permutation_scores.mean():.3f} +/- {permutation_scores.std():.3f}")
 
```

    P-value: 0.010
    Original Score: 0.947
    Permutation Scores: 0.333 +/- 0.039

