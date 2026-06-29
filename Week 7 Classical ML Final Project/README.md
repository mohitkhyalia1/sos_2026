# Week 7 Classical ML Final Project

Mohit Khyalia | Summer of Science 2026 | IIT Bombay

---

Week 7 Classical ML Final Project establishes the strongest possible classical ML baseline on the Spaceship Titanic dataset. No EDA, no preprocessing - both are complete from Week 5 Advanced Classical ML and Week 6 Deep Learning Infrastructure. Week 7 Classical ML Final Project is modelling, optimisation, and analysis only.

## Notebooks

- **spaceship_titanic_classical_ml.ipynb** - **(Major notebook)** Complete classical ML pipeline in one notebook: 5-fold CV baseline comparison across 6 classifiers, GridSearchCV tuning of the top 2 models, CV heatmaps, tuned vs default comparison, feature importance validation, threshold optimisation, final evaluation, and experiment summary table.

- **spaceship_titanic_error_analysis.ipynb** - Misclassification analysis of the best model: confusion matrix, precision-recall curve, optimal threshold derivation, ROC curve, error rate by demographic group (HomePlanet, Deck, CryoSleep, AgeGroup), highest-confidence error examination.

- **spaceship_titanic_submission.ipynb** - Best model fit on full labelled data, optimal threshold applied, Kaggle submission CSV generated, Week 7 Classical ML Final Project experiment summary table presented. Sets up the comparison table that Week 8 Deep Learning Final Project completes.

## Workflow

```text
Week 6 Deep Learning Infrastructure preprocessing
    |
[spaceship_titanic_classical_ml]
    Baseline comparison -> GridSearchCV tuning -> feature importance -> threshold analysis
    |
[spaceship_titanic_error_analysis]
    Misclassification patterns -> optimal threshold decision
    |
[spaceship_titanic_submission]
    Final model on full data -> Kaggle CSV (classical baseline)
```

## Key Takeaways

- CryoSleep and IsSpender are the strongest features - validated by feature importances against the Week 5 Advanced Classical ML EDA findings.
- GridSearchCV over Gradient Boosting hyperparameters (n_estimators, learning_rate, max_depth, subsample) produces meaningful gains over the default.
- The false positive / false negative breakdown by demographic group reveals which passenger types the model struggles with - informing potential additional feature engineering in Week 8 Deep Learning Final Project.

---
*Week 7 Classical ML Final Project - Mohit Khyalia, Summer of Science 2026, IIT Bombay*
