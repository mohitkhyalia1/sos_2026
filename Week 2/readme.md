# Week 02 — Data Preprocessing and Exploratory Data Analysis

Mohit Khyalia | Summer of Science 2026 | IIT Bombay

---

Missing values, scaling, encoding, train-test mechanics, class imbalance, and EDA, closing with a text-classification mini-project.

## Notebooks

- **handling_missing_values.ipynb** — isnull/dropna/fillna, mean/median/mode imputation
- **data_standardization.ipynb** — StandardScaler vs MinMaxScaler, distance and outlier comparisons
- **label_encoding.ipynb** — LabelEncoder vs one-hot, ordinal vs nominal categories
- **train_test_split.ipynb** — split ratios, random_state, stratify, data leakage demo
- **imbalanced_dataset.ipynb** — accuracy trap, oversampling, undersampling, class_weight
- **eda_practice.ipynb** — describe/info, distributions, countplots, correlation heatmap
- **spam_mail_prediction.ipynb** — TF-IDF + Logistic Regression on SMS spam/ham

## Key Takeaways

- Missing value handling changes the statistics of the data, not just the row count
- Feature scale determines which features a distance-based model can use
- LabelEncoder assigns numeric codes alphabetically, not meaningfully — check for a real order first
- Data leakage can look only slightly better, not obviously wrong
- Accuracy alone can hide a model that does nothing useful — check class balance first
- These checks carried over to text data with no real changes

## Relation to Spaceship Titanic

- Missing value approach (median/mode) carries over to Week 7
- HomePlanet / Destination have no natural order → one-hot, same as Property_Area here
- Spending columns are likely skewed/outlier-prone → same scaling caution as the income experiment
- Check "Transported" class balance before assuming it
- Fit any scaler/encoder on training data only

---
*Week 02 — Mohit Khyalia, Summer of Science 2026, IIT Bombay*
