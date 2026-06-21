# Week 04 — Supervised Learning and Model Evaluation

Mohit Khyalia | Summer of Science 2026 | IIT Bombay

---

Linear and Logistic Regression built from scratch, SVM basics, overfitting/underfitting, cross-validation, model evaluation, and applied machine learning workflows on real datasets.

## Notebooks

- **linear_regression_from_scratch.ipynb** — custom LinearRegression class, MSE, gradient descent, convergence plot, sklearn comparison
- **logistic_regression_from_scratch.ipynb** — sigmoid, log-loss, gradient descent, custom LogisticRegression class, confusion matrix, sklearn comparison
- **svm_basics.ipynb** — support vectors, margin, effect of C, linear vs RBF kernel, gamma effect
- **overfitting_underfitting_bias_variance.ipynb** — polynomial regression at varying complexity, train/test error curves, bias-variance via repeated sampling
- **cross_validation_and_model_selection.ipynb** — train/validation/test split, K-Fold CV, model comparison, basic GridSearchCV
- **evaluation_metrics.ipynb** — confusion matrix, accuracy/precision/recall/F1, ROC curve, threshold effects, imbalanced data example
- **house_price_prediction.ipynb** — Linear Regression on a housing dataset, MSE/R2, residual analysis
- **loan_status_prediction.ipynb** — missing values, encoding, scaling, stratified split, SVM classifier

## Key Takeaways

- Scratch implementations of Linear and Logistic Regression converge to nearly the same parameters as sklearn — the difference is mainly regularisation and solver efficiency
- A learning rate that's too high causes divergence, not just slow convergence
- SVM's C and RBF's gamma control the same underlying tradeoff as regularisation elsewhere — tighter fit vs wider margin
- Train/test error curves make overfitting visible as a numeric gap, not just a visual one
- K-Fold cross-validation gives a more stable performance estimate than any single train/validation split
- Accuracy alone can look fine even when a model is doing almost nothing useful — Precision, Recall, and the confusion matrix expose this
- Threshold choice trades Precision against Recall directly — 0.5 is a default, not a rule

---
*Week 04 — Mohit Khyalia, Summer of Science 2026, IIT Bombay*
