# Week 01 — Python and Data Science Basics

**Summer of Science 2026 | CS03: Artificial Intelligence and Machine Learning**  
**Mohit Khyalia | IIT Bombay**

---

## Objectives

By the end of Week 1, the goal was to be comfortable with:
- Core Python patterns that appear repeatedly in ML code
- NumPy arrays and vectorised operations as the foundation for all numerical computation
- Pandas DataFrames for loading, inspecting, and cleaning tabular data
- Matplotlib and Seaborn for creating visualizations used in EDA and reporting
- Completing the first two end-to-end supervised learning projects using scikit-learn

---

## Topics Covered

### Python Fundamentals (`python_basics.ipynb`)
- Data types and type conversion — why `int()` truncates rather than rounds
- Lists: slicing, comprehensions, the copy gotcha (reference vs `.copy()`)
- Tuples: immutability, unpacking — connecting to `.shape` in NumPy
- Sets: O(1) lookup, set operations for comparing feature lists
- Dictionaries: comprehensions, `.get()`, iterating with `.items()`
- Control flow: `if/elif/else`, `enumerate()`, `zip()`
- Functions: default arguments, docstrings, `*args`/`**kwargs`
- String operations for cleaning column names
- Experiment: list vs set lookup timing on 1 million elements

### NumPy (`numpy_basics.ipynb`)
- Array creation: `zeros`, `ones`, `arange`, `linspace`, `eye`
- Vectorised operations vs Python loops — performance comparison on 1M elements
- Indexing and slicing: `X[:, :-1]` pattern for feature/label separation
- Boolean indexing for filtering rows
- Mathematical operations: dot product, matrix multiplication, column-wise statistics
- Manual StandardScaler implementation to understand what it does internally
- Manual MSE computation
- Broadcasting: why `X - means` works without explicit loops
- Random number generation with `np.random.default_rng(seed)`
- Shape rules: `(n,)` vs `(n,1)` and when each matters for sklearn

### Pandas (`pandas_basics.ipynb`)
- Creating DataFrames from dictionaries
- `info()` and `describe()` — the two commands to always run first
- Missing value summary: `isnull().sum()` with percentage
- Selection: `[]`, `.loc[]`, `.iloc[]` — when to use each
- Boolean filtering and `.query()`
- Missing value handling: `dropna()` vs `fillna()` with median/mode
- Feature engineering: derived columns, `pd.cut()` for binning
- `groupby().agg()` for class-conditional statistics
- `.value_counts()` for categorical distributions
- Converting to NumPy with `.values` before passing to sklearn
- Mini-exercise: `dataset_report()` function for standardised EDA output

### Matplotlib (`matplotlib_basics.ipynb`)
- Object-oriented API: `fig, ax = plt.subplots()` vs `plt.plot()`
- Line plots: training/validation loss curves with annotation
- Scatter plots: two-class separation visualisation
- Histograms: feature distribution analysis; symmetric vs skewed
- Bar charts: grouped metric comparison across models
- Subplots: building multi-panel EDA dashboards
- Gradient descent visualisation: connecting the math to the plot
- Saving figures: `plt.savefig()` with `dpi=150` for report quality

### Seaborn (`seaborn_basics.ipynb`)
- `countplot` with `hue`: categorical feature vs target variable
- `histplot` with `kde=True`: distribution comparison by class
- `heatmap`: correlation matrix with `annot=True`
- `boxplot` with `hue`: group comparison across two categorical dimensions
- `pairplot` with `hue`: all pairwise relationships coloured by class
- Seaborn + Matplotlib combined: sharing a legend across a multi-panel figure
- When to use Seaborn vs Matplotlib directly

---

## Mini-Projects

### Project 1: Rock vs Mine Prediction (`rock_vs_mine_prediction.ipynb`)

**Dataset:** SONAR dataset — 208 samples, 60 features (sonar frequency band energies)  
**Task:** Predict whether a sonar return is from a rock (R) or a metal mine (M)  
**Model:** Logistic Regression

**What was studied and learned:**
- Mean signal profiles for each class — mines and rocks have different shapes across frequency bands
- High feature-to-sample ratio (60 features / 208 samples) makes overfitting likely
- Training accuracy: ~85.6% | Test accuracy: ~76.2% | Gap: ~9.4% indicating mild overfitting
- Traced a single prediction end-to-end: 60-dimensional input → dot product → sigmoid → class label
- Investigated the effect of the regularisation parameter C: lower C reduces overfitting
- Classification report interpretation: in a real mine-detection setting, Recall for class M matters most

**Key takeaway:** High-dimensional small datasets require attention to overfitting. The regularisation parameter C in Logistic Regression directly controls the bias-variance trade-off.

---

### Project 2: Diabetes Prediction (`diabetes_prediction.ipynb`)

**Dataset:** Pima Indians Diabetes dataset — 768 samples, 8 clinical features  
**Task:** Predict whether a patient has diabetes (1) or not (0)  
**Model:** Support Vector Machine (linear kernel)

**What was studied and learned:**
- Zero-value problem: Insulin (48.7% zeros) and SkinThickness (29.6% zeros) are effectively missing data encoded as 0 — a real data quality issue
- Feature range analysis: Insulin range is ~846 while DiabetesPedigreeFunction range is ~2.3 — without scaling, SVM ignores the small-range features
- Explicitly compared SVM with scaling (~76.6% test accuracy) vs without scaling (~73.4%) to confirm the importance of StandardScaler
- Critical data leakage rule: scaler must be `fit` on training data only, then `transform` applied to test
- Recall for diabetic class is only ~57%: 43% of diabetic patients predicted as healthy — the most concerning type of error in a medical context
- Comparison with Logistic Regression: both linear models achieve similar accuracy (~77%) on this data

**Key takeaway:** Feature scaling is not optional for distance-based models. The most important evaluation metric depends on the problem context — in healthcare, minimising false negatives (missed diagnoses) often matters more than overall accuracy.

---

## Skills Learned

**Programming:**
- Python comprehensions, unpacking, and functional patterns used in ML code
- NumPy vectorisation, broadcasting, and efficient array operations
- Pandas EDA workflow: `info()` → `describe()` → `isnull().sum()` → `groupby()` → `value_counts()`

**Machine Learning Concepts:**
- End-to-end supervised classification workflow: load → EDA → split → scale → fit → evaluate
- Why feature scaling is mandatory for distance-based algorithms (SVM) but less critical for tree-based or linear models
- The train-test split as the fundamental validation methodology
- The meaning of training accuracy vs test accuracy and their gap as an overfitting signal
- Interpreting classification reports: Precision, Recall, F1-score, and when each matters
- The regularisation parameter C as a lever between underfitting and overfitting

**Visualization:**
- Overlapping histograms with `alpha` for class comparison
- Mean signal profile plots for understanding class structure
- Confusion matrix heatmaps with per-cell interpretation
- Subplots for dashboard-style EDA layouts

---

## Key Takeaways

1. **Always run `info()` and `describe()` first.** They reveal missing values, data types, and suspicious zero values before any modelling starts.

2. **Feature scaling changes SVM results meaningfully.** The ~3 percentage point improvement from scaling on the diabetes dataset demonstrates this concretely.

3. **Training accuracy and test accuracy are both necessary.** Reporting only one of them gives an incomplete picture — a large gap indicates overfitting, a small gap indicates the model generalises reasonably.

4. **The right metric depends on the problem.** Overall accuracy is fine for balanced datasets; Recall matters more when false negatives are costly (healthcare, fraud detection, mine detection).

5. **`scaler.fit_transform(X_train)` then `scaler.transform(X_test)`.** This exact pattern prevents data leakage. Fitting the scaler on test data would inflate accuracy estimates.

---

## Folder Contents

```
Week-01-Python-and-Data-Science-Basics/
│
├── python_basics.ipynb          # Python fundamentals with ML-relevant examples
├── numpy_basics.ipynb           # NumPy operations, vectorisation, and MSE gradient
├── pandas_basics.ipynb          # DataFrame operations, EDA workflow, missing values
├── matplotlib_basics.ipynb      # Visualization for ML: loss curves, scatter, histograms
├── seaborn_basics.ipynb         # Statistical plots: countplot, heatmap, boxplot, pairplot
├── rock_vs_mine_prediction.ipynb # Project 1: SONAR classification with Logistic Regression
├── diabetes_prediction.ipynb    # Project 2: Diabetes classification with SVM + scaling demo
└── README.md                    # This file
```

---

## Resources Used

- Siddhardhan, *Complete Machine Learning Course in 60 Hours*, Parts 1–2 (YouTube + GitHub)
- scikit-learn documentation: `LogisticRegression`, `SVC`, `StandardScaler`, `train_test_split`
- NumPy documentation: array operations, broadcasting rules
- Pandas documentation: `DataFrame.groupby`, `DataFrame.describe`
- SONAR dataset: UCI ML Repository
- Pima Indians Diabetes dataset: UCI ML Repository / NIDDK

---

*Week 01 — Mohit Khyalia | Summer of Science 2026 | IIT Bombay*
