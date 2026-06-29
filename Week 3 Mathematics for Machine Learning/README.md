# Week 03 — Mathematics for Machine Learning

Mohit Khyalia | Summer of Science 2026 | IIT Bombay

---

Vectors, matrices, gradient descent, and statistics/probability — the math layer underneath the models studied from Week 4 onward.

## Notebooks

- **vector_operations.ipynb** — addition, dot product, norm, cosine similarity, projection, 1D vs 2D arrays, timing check
- **matrix_operations.ipynb** — addition, transpose, multiplication, broadcasting, rank, determinant, inverse
- **gradient_descent_visualization.ipynb** — cost curve, update rule, learning rate comparison, single-variable linear regression demo
- **statistics_probability_basics.ipynb** — mean/median/mode, variance/std, covariance/correlation, conditional probability, Bayes' theorem, normal distribution

## Key Takeaways

- Matrix multiplication is the dot product applied across every row/column pair at once
- Broadcasting (`data - col_means`) is the actual mechanism behind StandardScaler
- A learning rate that's too high causes the cost to diverge instead of converge — same failure seen in the Week 4 from-scratch Linear Regression notebook
- Mean vs median diverges on skewed data — the same check used for imputation decisions in Week 2
- A rare condition with a high-sensitivity test can still have a low P(condition | positive) — base rates matter, same reasoning as why Recall alone doesn't tell the full story on imbalanced data
- Standard deviation shows up in three places this week: variance formulas, the normal distribution's empirical rule, and StandardScaler's formula from Week 2

---
*Week 03 — Mohit Khyalia, Summer of Science 2026, IIT Bombay*
