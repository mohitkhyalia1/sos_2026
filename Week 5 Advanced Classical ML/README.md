# Week 05 — Advanced Classical Machine Learning

Mohit Khyalia | Summer of Science 2026 | IIT Bombay

---

Advanced classical ML techniques applied to loan and house price datasets, with the Spaceship Titanic dataset introduced through a full EDA. Every notebook in this week produces a workflow or insight reused directly in Weeks 6–8.

## Notebooks

- **decision_trees_and_ensembles.ipynb** — Gini impurity, tree depth vs accuracy, pruning, Random Forest (bagging), Gradient Boosting (boosting), feature importances. Applied to the loan status dataset. The CV comparison loop and feature importance visualization are reused verbatim in Week 7.

- **regularization_and_feature_selection.ipynb** — Ridge and Lasso shrinkage paths, cross-validated alpha selection, Lasso zero-coefficient feature selection. Applied to the house price dataset. Connects to `alpha` in MLPClassifier (Week 6) and `kernel_regularizer` in Keras (Week 8).

- **feature_engineering_for_spaceship_titanic.ipynb** — Aggregation features, interaction features, binning, polynomial expansion, Cabin-style string parsing (`Deck/Number/Side`), one-hot vs label encoding comparison. Every technique here is implemented in `spaceship_titanic_preprocessing.ipynb` (Week 6).

- **machine_learning_pipelines.ipynb** — Data leakage demonstration, `sklearn` Pipeline, ColumnTransformer for mixed numeric/categorical data, GridSearchCV over the full pipeline. The ColumnTransformer structure is reused in every Spaceship Titanic modelling notebook.

- **spaceship_titanic_eda.ipynb** — Full EDA of the Kaggle dataset: missing value audit, target distribution, Cabin parsing, spending feature distributions, CryoSleep × spending interaction, correlation analysis, age distribution. Concludes with a Feature Engineering Decisions table that drives the Week 6 preprocessing pipeline.

## Key Takeaways

- Gradient Boosting and Random Forest consistently outperform single trees — both are the primary candidates for Week 7.
- Lasso performs feature selection implicitly by zeroing weak coefficients — confirmed by tree-based importance rankings.
- Manual preprocessing before CV introduces data leakage; Pipeline is the correct solution and is used in all subsequent work.
- The CryoSleep × spending interaction is the strongest signal in the Spaceship Titanic dataset.

---
*Week 05 — Mohit Khyalia, Summer of Science 2026, IIT Bombay*
