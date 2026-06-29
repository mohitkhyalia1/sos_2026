# Week 08 — Deep Learning Final Project

Mohit Khyalia | Summer of Science 2026 | IIT Bombay

---

Week 8 is the flagship week and the final notebook of the repository. It applies deep learning to the Spaceship Titanic dataset, systematically extends the Week 6 Keras configuration, and fairly benchmarks the neural network against the best classical model from Week 7.

## Notebook

- **spaceship_titanic_deep_learning.ipynb** — **(Flagship notebook)** Complete deep learning pipeline:
  - **Part 1:** Baseline — Week 6 configuration applied to Spaceship Titanic (~8700 samples vs ~490 in Week 6).
  - **Part 2:** Architecture search — 1 to 4 hidden layers, finding the depth at which returns diminish.
  - **Part 3:** BatchNormalization — compared against Dropout-only and combined configurations.
  - **Part 4:** Optimizer comparison — Adam vs RMSprop vs SGD+momentum on Spaceship Titanic data.
  - **Part 5:** Learning rate scheduling — `ReduceLROnPlateau` vs fixed learning rate.
  - **Part 6:** Final model assembly — best configuration from all experiments.
  - **Part 7:** Classical vs DL head-to-head — same validation fold, same metrics.
  - **Part 8:** ROC curve comparison — threshold-independent evaluation.
  - **Part 9:** Final Kaggle submission — best model fit on full labelled data.

## Repository Conclusion

This notebook concludes an 8-week progression from Python fundamentals to a production-grade ML pipeline. Every design decision made in Week 8 is traceable to a prior notebook: the optimizer choice to `deep_learning_keras.ipynb`, the feature engineering to `spaceship_titanic_eda.ipynb`, the preprocessing to `spaceship_titanic_preprocessing.ipynb`, and the classical baseline to `spaceship_titanic_classical_ml.ipynb`.

The repository demonstrates feature engineering, reusable preprocessing workflows, systematic experimentation, model comparison, error analysis, neural network optimisation, and engineering maturity across a single continuous project rather than isolated weekly assignments.

---
*Week 08 — Mohit Khyalia, Summer of Science 2026, IIT Bombay*
