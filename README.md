# Machine Learning Foundations - Summer of Science 2026

This repository documents my Summer of Science 2026 work at IIT Bombay. It is an 8-week machine learning path that starts with Python and data science fundamentals, then moves through preprocessing, mathematical foundations, supervised learning, advanced classical ML, neural network infrastructure, and a final end-to-end Spaceship Titanic project.

The work is organized as Jupyter notebooks with explanations, experiments, visualizations, observations, and project-style workflows. The later weeks intentionally reuse decisions from earlier weeks, so the repository reads as one continuous progression rather than a set of isolated assignments.

## Repository Snapshot

- **40 Jupyter notebooks** across 8 weekly modules
- **8 CSV datasets** stored in `dataset/`
- **2 submitted reports** in `Reports/`
- Classical ML and deep learning workflows using scikit-learn, TensorFlow/Keras, pandas, NumPy, Matplotlib, and Seaborn
- Final applied project on Kaggle's Spaceship Titanic dataset with preprocessing, feature engineering, model comparison, error analysis, and submission generation

## Repository Structure

```text
sos_2026/
|-- Week 1 Python & Data Science Fundamentals/
|-- Week 2 Data Preprocessing & EDA/
|-- Week 3 Mathematics for Machine Learning/
|-- Week 4 Supervised Learning & Model Evaluation/
|-- Week 5 Advanced Classical ML/
|-- Week 6 Deep Learning Infrastructure/
|-- Week 7 Classical ML Final Project/
|-- Week 8 Deep Learning Final Project/
|-- dataset/
|-- Reports/
`-- README.md
```

## Weekly Roadmap

| Week | Focus | Main Output |
| --- | --- | --- |
| Week 1 Python & Data Science Fundamentals | Python, NumPy, pandas, Matplotlib, Seaborn | First ML mini-projects on sonar and diabetes data |
| Week 2 Data Preprocessing & EDA | Preprocessing, EDA, encoding, scaling, splitting, class imbalance | Spam mail text classification pipeline |
| Week 3 Mathematics for Machine Learning | Linear algebra, gradient descent, statistics, probability | Mathematical grounding for later models |
| Week 4 Supervised Learning & Model Evaluation | Supervised learning and model evaluation | Scratch implementations, SVMs, CV, metrics, and applied prediction projects |
| Week 5 Advanced Classical ML | Advanced classical ML | Trees, ensembles, regularization, feature selection, pipelines, and Spaceship Titanic EDA |
| Week 6 Deep Learning Infrastructure | Deep learning infrastructure | Neural network foundations, sklearn MLP, Keras experiments, and reusable Spaceship preprocessing |
| Week 7 Classical ML Final Project | Classical ML final project | Tuned classical Spaceship Titanic model, error analysis, and submission workflow |
| Week 8 Deep Learning Final Project | Deep learning final project | Neural network experimentation and fair comparison against the classical baseline |

## Detailed Week Breakdown

### Week 1 Python & Data Science Fundamentals

Introduces the core Python stack used throughout the repository:

- Python basics, control flow, functions, and notebook workflows
- NumPy arrays, vectorized operations, broadcasting, reshaping, and simple gradient calculations
- pandas DataFrames, filtering, grouping, sorting, missing values, and handoff to scikit-learn
- Matplotlib and Seaborn for ML-oriented visualization
- Mini-projects:
  - Rock vs Mine prediction using Logistic Regression on sonar data
  - Diabetes prediction using SVM and Logistic Regression

### Week 2 Data Preprocessing & EDA

Builds the practical data preparation layer:

- Missing-value detection, dropping, and mean/median/mode imputation
- Standardization and MinMax scaling
- Label encoding, one-hot encoding, and ordinal category handling
- Train-test splitting, stratification, random states, and leakage awareness
- Imbalanced dataset handling with resampling and class weights
- Exploratory data analysis with distributions, count plots, and correlation heatmaps
- Mini-project:
  - Spam/Ham mail prediction using TF-IDF features and Logistic Regression

### Week 3 Mathematics for Machine Learning

Connects ML code to the mathematics behind it:

- Vector operations, dot products, norms, cosine similarity, and projection
- Matrix operations, transpose, multiplication, broadcasting, rank, determinant, and inverse
- Gradient descent visualization with learning-rate comparisons
- Statistics and probability: mean, median, variance, covariance, correlation, conditional probability, Bayes' theorem, and normal distribution

### Week 4 Supervised Learning & Model Evaluation

Turns the earlier foundations into supervised learning workflows:

- Linear Regression from scratch with MSE, gradient descent, convergence plots, and sklearn comparison
- Logistic Regression from scratch with sigmoid, log-loss, gradient descent, confusion matrix, and sklearn comparison
- SVM basics: margins, support vectors, C, linear kernel, RBF kernel, and gamma
- Overfitting, underfitting, and bias-variance using polynomial regression
- Train/validation/test split, K-Fold CV, model comparison, and GridSearchCV
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC curve, threshold tuning, and imbalanced-data examples
- Mini-projects:
  - Heart disease prediction with Logistic Regression
  - House price prediction with Linear Regression and residual analysis
  - Loan status prediction with preprocessing and SVM classification

### Week 5 Advanced Classical ML

Introduces stronger classical ML tools and starts the final project dataset:

- Decision Trees, Gini impurity, tree depth, pruning, Random Forest, Gradient Boosting, and feature importances
- Ridge and Lasso regularization, cross-validated alpha selection, and Lasso-based feature selection
- Spaceship Titanic feature engineering: Cabin parsing, spending aggregates, interaction features, binning, polynomial features, and encoding comparisons
- Pipeline and ColumnTransformer workflows to avoid data leakage
- Full Spaceship Titanic EDA covering missingness, target balance, spending behavior, CryoSleep interactions, Cabin structure, age distribution, and correlation patterns

### Week 6 Deep Learning Infrastructure

Builds the neural-network and preprocessing foundation used by the final week:

- Single neuron as Logistic Regression, activation functions, vanishing gradients, XOR failure, and a two-layer NumPy network
- MLPClassifier experiments with architecture, activation functions, L2 regularization, and GridSearchCV
- Keras Sequential models with Dropout, EarlyStopping, optimizer comparisons, learning-rate effects, batch-size experiments, and dropout tuning
- Complete Spaceship Titanic preprocessing pipeline with:
  - Cabin parsing into Deck, Number, and Side
  - Spending aggregation and `IsSpender`
  - Age binning
  - Log transforms for skewed spending columns
  - ColumnTransformer-based numeric and categorical preprocessing

### Week 7 Classical ML Final Project

Builds the strongest classical baseline for Spaceship Titanic:

- Baseline comparison across multiple classifiers with 5-fold cross-validation
- GridSearchCV tuning of the strongest models
- CV heatmaps, tuned-versus-default comparison, and feature importance validation
- Threshold optimization and final evaluation
- Misclassification analysis using confusion matrix, precision-recall curve, ROC curve, demographic error breakdown, and high-confidence error inspection
- Final classical model fit on full labelled data and Kaggle submission generation

### Week 8 Deep Learning Final Project

Completes the repository with a deep learning version of the final project:

- Baseline Keras model using the Week 6 Deep Learning Infrastructure recommended configuration
- Architecture search across 1 to 4 hidden layers
- BatchNormalization, Dropout, and combined regularization comparisons
- Optimizer comparison: Adam, RMSprop, and SGD with momentum
- Learning-rate scheduling with `ReduceLROnPlateau`
- Final neural network assembly from the best experimental settings
- Classical ML versus deep learning comparison on the same validation fold
- ROC curve comparison and final Kaggle submission workflow

## Final Project Flow

The Spaceship Titanic work is the main through-line of the second half of the repository:

```text
Week 5 Advanced Classical ML
  EDA and feature engineering decisions
      |
Week 6 Deep Learning Infrastructure
  Reusable preprocessing pipeline and deep learning setup
      |
Week 7 Classical ML Final Project
  Tuned classical ML baseline, thresholding, error analysis, submission
      |
Week 8 Deep Learning Final Project
  Deep learning experiments, classical-vs-DL comparison, final submission
```

## Datasets

The `dataset/` directory contains all CSV data used across the notebooks.

| Dataset | Rows | Columns | Used For |
| --- | ---: | ---: | --- |
| `sonar data.csv` | 208 | 61 | Rock vs Mine classification |
| `diabetes.csv` | 768 | 9 | Diabetes prediction |
| `spam_mail_data.csv` | 5,572 | 2 | Spam/Ham text classification |
| `heart_disease_data.csv` | 1,025 | 14 | Heart disease classification |
| `house_price_data.csv` | 1,460 | 81 | House price regression |
| `loan_status_data.csv` | 614 | 13 | Loan approval classification |
| `spaceship_titanic_train.csv` | 8,693 | 14 | Spaceship Titanic training and validation |
| `spaceship_titanic_test.csv` | 4,277 | 13 | Spaceship Titanic final submission |

## Mini-Projects and Applied Work

- **Rock vs Mine Prediction** - Logistic Regression classification on sonar signal features
- **Diabetes Prediction** - SVM and Logistic Regression comparison with feature scaling
- **Spam Mail Prediction** - TF-IDF vectorization and Logistic Regression for SMS spam detection
- **Heart Disease Prediction** - Logistic Regression classification using clinical features
- **House Price Prediction** - Regression workflow with preprocessing, Linear Regression, and residual analysis
- **Loan Status Prediction** - Missing-value handling, encoding, feature scaling, and SVM classification
- **Spaceship Titanic Classical ML** - Feature engineering, tuned ensemble models, threshold optimization, error analysis, and Kaggle submission
- **Spaceship Titanic Deep Learning** - Keras model experiments, architecture search, optimizer/scheduler comparison, and classical-vs-DL benchmarking

## Skills Practiced

- Python programming and notebook-based experimentation
- NumPy vectorization and matrix operations
- pandas data cleaning, transformation, and analysis
- Visualization with Matplotlib and Seaborn
- Exploratory Data Analysis
- Missing-value handling, scaling, encoding, stratified splitting, and leakage prevention
- Linear algebra, statistics, probability, and gradient descent
- Linear Regression and Logistic Regression from scratch
- Classification and regression using scikit-learn
- SVMs, decision trees, Random Forest, Gradient Boosting, Ridge, Lasso, and feature selection
- Pipeline and ColumnTransformer design for reproducible preprocessing
- Cross-validation, GridSearchCV, threshold tuning, and model comparison
- Evaluation with confusion matrices, precision, recall, F1-score, ROC curves, MSE, MAE, R2, and residual analysis
- Neural network basics, MLPClassifier, Keras Sequential models, Dropout, BatchNormalization, EarlyStopping, and learning-rate scheduling
- Error analysis, experiment tracking tables, and final submission generation

## How to Run

1. Clone this repository.
2. Open the folder in VS Code, JupyterLab, or another Jupyter notebook environment.
3. Create and activate a Python environment.
4. Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow jupyter
```

5. Launch Jupyter:

```bash
jupyter notebook
```

6. Open any weekly folder and run the notebooks in order.

Most concept notebooks either generate synthetic data inside the notebook or load CSV files from `dataset/`. The Spaceship Titanic final project is most coherent when followed in order from Week 5 Advanced Classical ML through Week 8 Deep Learning Final Project.

## Reports

The `Reports/` directory contains the submitted Summer of Science documents:

- `SOS_POA_24B2289.pdf` - Plan of Action
- `Midterm_Report_SOS_24B2289.pdf` - Midterm Progress Report

## Author

**Mohit Khyalia**  
B.Tech. Mechanical Engineering  
Indian Institute of Technology Bombay  
Summer of Science 2026
