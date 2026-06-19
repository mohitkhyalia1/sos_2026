# Week 02 — Data Preprocessing and Exploratory Data Analysis

**Summer of Science 2026 | CS03: Artificial Intelligence and Machine Learning**  
**Mohit Khyalia | IIT Bombay**

---

## Week Overview

Where Week 1 was about getting comfortable with Python and the basic libraries, Week 2 is where I actually had to deal with the fact that real data is messy. Every dataset I'd touched in Week 1 was already mostly clean and ready to go — this week I worked through the things that have to happen *before* that point: missing values, scale mismatches between features, converting categories into numbers without misleading the model, splitting data properly, dealing with imbalanced classes, and doing a genuine first-look EDA on a dataset rather than rushing straight to modelling.

The week closes out with the Spam Mail Prediction mini-project, which pulls together pretty much everything from earlier in the week and applies it to text data for the first time — a useful test of whether I'd actually understood these ideas generally, or just for the specific kind of tabular data I'd been working with so far.

---

## Topics Covered

### Handling Missing Values (`handling_missing_values.ipynb`)
- Detecting missing values with `isnull()` and `isnull().sum()`
- Visualizing where missing values are concentrated using a heatmap
- `dropna()` and the real cost of using it carelessly (lost 45% of a small synthetic dataset)
- `dropna(thresh=n)` as a gentler middle ground
- `fillna()` with mean, median, and mode, and reasoning about which to use based on skew
- A small experiment comparing fill strategies, including a "fill with zero" example to show why a meaningless constant is a bad default

### Data Standardization and Scaling (`data_standardization.ipynb`)
- Why feature scale differences distort distance-based calculations — demonstrated with an actual Euclidean distance breakdown (one feature contributing 100% of the distance before scaling)
- `StandardScaler`: formula, implementation, and verification against population vs sample standard deviation
- `MinMaxScaler`: formula and implementation
- A side-by-side outlier experiment showing MinMaxScaler is more fragile to extreme values than StandardScaler

### Encoding Categorical Variables (`label_encoding.ipynb`)
- `LabelEncoder` and the discovery that it assigns numbers alphabetically, not meaningfully
- A concrete demonstration of how Label Encoding can imply a false "distance" between unordered categories
- One-hot encoding via `pd.get_dummies()`, including the `drop_first` parameter
- The distinction between ordinal categories (where Label Encoding with a deliberate order makes sense) and nominal categories (where one-hot is the safer choice)

### Train-Test Split (`train_test_split.ipynb`)
- Why evaluating a model on its own training data is misleading
- A deliberately constructed data leakage demonstration — fitting a scaler on the full dataset vs training data only, and comparing the (subtly) inflated accuracy
- Comparing different split ratios and the tradeoff between test set size and accuracy estimate stability
- What `random_state` actually controls, confirmed by testing reproducibility directly
- `stratify` and why it matters more as class imbalance increases

### Handling Imbalanced Datasets (`imbalanced_dataset.ipynb`)
- The "accuracy trap": a model that always predicts the majority class scoring 95% accuracy while doing nothing useful
- Oversampling the minority class and the precision/recall tradeoff it introduces
- Undersampling the majority class and the real cost of discarding data
- `class_weight='balanced'` as a way to rebalance without touching the dataset
- All three approaches compared side by side on the same fraud-detection-style dataset

### EDA Practice (`eda_practice.ipynb`)
- A complete first-look EDA workflow on a previously unseen dataset, from `info()`/`describe()` through to a written summary
- Numeric distributions with mean/median comparison to spot skew
- Categorical counts split by target using `sns.countplot()` with `hue`
- Class balance checking as a default first step, not an afterthought
- Correlation heatmap and checking whether the correlations match the actual relationships built into the data
- Practicing writing a short interpretation after every plot rather than just generating visuals

### Spam Mail Prediction (`spam_mail_prediction.ipynb`)
- End-to-end project applying the week's preprocessing lessons to text data for the first time
- Basic text cleaning (lowercasing, stripping non-alphabetic characters)
- TF-IDF vectorization — understanding it as one column per vocabulary word rather than a single numeric encoding
- Train-test split applied to text, with the same fit-on-training-data-only rule from earlier in the week
- Logistic Regression on TF-IDF features, evaluated with Accuracy, Precision, Recall, and F1
- `class_weight='balanced'` applied again, this time to text, to see if the imbalance lesson transfers
- Manual testing on hand-written example messages as an extra sanity check beyond the test set metrics

---

## Skills Learned

**Preprocessing:**
- A working checklist for any new dataset: check missing values → check types → check scale → check encoding needs → check class balance
- Concrete understanding of *why* each preprocessing step matters, not just the syntax to run it
- The specific mechanics of data leakage and how to avoid it by being careful about when `.fit()` is called

**Evaluation:**
- Why accuracy alone can be actively misleading, demonstrated rather than just stated
- Reading Precision, Recall, and F1 together to understand what kind of errors a model is making
- Comparing multiple imbalance-handling strategies on the same data to see their actual tradeoffs rather than assuming one is universally best

**Text-specific:**
- TF-IDF as a way of converting variable-length text into fixed-length numeric vectors
- That high-dimensional sparse data (1000+ TF-IDF features) needs a different kind of intuition-building than small tabular datasets — checking top-weighted words and testing example inputs, rather than just eyeballing a table

---

## Key Takeaways

1. **Missing value handling is a decision, not a checkbox.** Comparing `dropna()` against `fillna()` side by side on the same data showed that deletion isn't neutral — it shifts the statistics of whatever data is left.

2. **Scaling changes which features a model can actually use.** Computing the literal percentage contribution to a distance calculation made this concrete in a way that just reading "scaling matters for SVM" never did.

3. **Label Encoding silently assumes an order that might not exist.** This is an easy mistake to make by just running the same code on every categorical column without checking whether it actually applies.

4. **Data leakage doesn't always look obviously wrong.** The leaked version of the pipeline in `train_test_split.ipynb` only looked a little better than the correct version — which is arguably more dangerous than an obviously broken result, since it's much easier to miss.

5. **Accuracy can hide a model that's doing almost nothing useful.** The "always predict majority class" experiment in `imbalanced_dataset.ipynb` is the single most useful thing from this week — a 95% accuracy with zero actual learning.

6. **These principles aren't tabular-data-specific.** Applying class balance checks, the leakage rule, and metric comparisons to text data in `spam_mail_prediction.ipynb` worked the same way as it did for numeric data, which suggests these are general ML principles rather than tricks tied to one data type.

---

## Relation to the Spaceship Titanic Project

The Project Brief describes the Spaceship Titanic dataset as having missing values across both categorical and numeric columns, a mix of column types that will need different encoding strategies, and spending columns (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck) that are very likely to be skewed given how spending data usually looks — all of which this week was direct practice for. Specifically:

- The missing value imputation approach from `handling_missing_values.ipynb` (median for numeric, mode for categorical) is the same approach I'll start with on Spaceship Titanic
- The Label Encoding vs One-Hot decision from `label_encoding.ipynb` applies directly to HomePlanet and Destination, which are unordered categories with no natural ranking
- The skew-checking habit from `data_standardization.ipynb` and `eda_practice.ipynb` is relevant given the spending columns are likely to have a long tail of high spenders, similar to the outlier experiment with income
- The class balance check from `imbalanced_dataset.ipynb` is something I now plan to run on the "Transported" column as the very first step in Week 7, rather than assuming it's balanced
- The data leakage discipline from `train_test_split.ipynb` — fit any scaler or encoder on training data only — applies to every preprocessing step I'll use on the final dataset

This week felt less like learning new syntax and more like building a checklist I'll actually reuse. Going into Week 3's math content next, the plan is to keep applying this same checklist as a starting point on every new dataset, the same way I did unprompted in `eda_practice.ipynb` and `spam_mail_prediction.ipynb` by the end of this week.

---

## Folder Contents

```
Week-02-Data-Preprocessing-and-EDA/
│
├── handling_missing_values.ipynb   # isnull, dropna, fillna with mean/median/mode comparison
├── data_standardization.ipynb      # StandardScaler vs MinMaxScaler, distance demonstration
├── label_encoding.ipynb            # LabelEncoder vs one-hot, ordinal vs nominal categories
├── train_test_split.ipynb          # Split ratios, random_state, stratify, data leakage demo
├── imbalanced_dataset.ipynb        # Accuracy trap, oversampling, undersampling, class_weight
├── eda_practice.ipynb              # Full first-look EDA workflow on an unseen dataset
├── spam_mail_prediction.ipynb      # End-to-end text classification mini-project (TF-IDF + Logistic Regression)
└── README.md                       # This file
```

---

## Resources Used

- Machine learning video lecture series and accompanying implementation exercises, used as the primary study resource for this week
- scikit-learn documentation: `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `train_test_split`, `TfidfVectorizer`, `LogisticRegression`
- Pandas documentation: `isnull`, `dropna`, `fillna`, `groupby`
- SMS Spam Collection dataset (UCI Machine Learning Repository)

---

*Week 02 — Mohit Khyalia | Summer of Science 2026 | IIT Bombay*
