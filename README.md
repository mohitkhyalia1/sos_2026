# Machine Learning Foundations - Summer of Science 2026

This repository contains my Summer of Science 2026 work at IIT Bombay, focused on building a practical foundation in machine learning. The work progresses from Python and data-handling basics to preprocessing, mathematical foundations, supervised learning, model evaluation, and applied mini-projects.

The repository is organized week by week. Each week contains Jupyter notebooks with explanations, code experiments, visualizations, and short reflections.

## Repository Structure

```text
sos_2026/
|-- Week 1/      Python, NumPy, pandas, visualization, first ML projects
|-- Week 2/      Data preprocessing, EDA, splitting, imbalance, spam detection
|-- Week 3/      Linear algebra, gradient descent, statistics, probability
|-- Week 4/      Supervised learning, validation, metrics, applied prediction
|-- dataset/     CSV datasets used across notebooks
|-- Reports/     Summer of Science proposal and midterm report PDFs
`-- README.md    Project overview
```

## Weekly Progress

### Week 1 - Python, Data Libraries, and First Models

Introduces the core Python tools used throughout the project:

- Python data types, control flow, functions, and basic performance checks
- NumPy arrays, vectorized operations, broadcasting, reshaping, and manual MSE gradient
- pandas DataFrames, filtering, missing values, grouping, sorting, and sklearn handoff
- Matplotlib and seaborn for ML-oriented visualization
- Mini-projects:
  - Rock vs Mine prediction using Logistic Regression on sonar data
  - Diabetes prediction using SVM and Logistic Regression

### Week 2 - Preprocessing and Exploratory Data Analysis

Covers the practical steps needed before model training:

- Missing-value detection and imputation
- Standardization and MinMax scaling
- Label encoding, one-hot encoding, and ordinal mapping
- Train-test split, stratification, random states, and data leakage
- Imbalanced datasets, oversampling, undersampling, and class weights
- Exploratory data analysis with distributions, count plots, and correlation heatmaps
- Mini-project:
  - Spam mail prediction using TF-IDF features and Logistic Regression

### Week 3 - Mathematics for Machine Learning

Builds the mathematical layer behind the models:

- Vector operations, dot product, norms, cosine similarity, and projection
- Matrix operations, transpose, multiplication, broadcasting, rank, determinant, and inverse
- Gradient descent visualization with learning-rate comparison
- Statistics and probability: mean, median, variance, covariance, correlation, conditional probability, Bayes' theorem, and normal distribution

### Week 4 - Supervised Learning and Model Evaluation

Connects the earlier preprocessing and math concepts to supervised learning workflows:

- Linear Regression from scratch with gradient descent and sklearn comparison
- Logistic Regression from scratch with sigmoid, log-loss, confusion matrix, and sklearn comparison
- SVM basics: margins, support vectors, C, linear kernel, RBF kernel, and gamma
- Overfitting, underfitting, and bias-variance through polynomial regression
- Cross-validation, train-validation-test split, K-Fold CV, model comparison, and GridSearchCV
- Evaluation metrics: accuracy, precision, recall, F1, ROC curve, and threshold tuning
- Mini-projects:
  - House price prediction using Linear Regression
  - Loan status prediction using SVM

## Datasets

The `dataset/` directory contains the local CSV files used by the notebooks:

| Dataset | Rows | Columns | Used For |
| --- | ---: | ---: | --- |
| `sonar data.csv` | 207 | 61 | Rock vs mine classification |
| `diabetes.csv` | 768 | 9 | Diabetes prediction |
| `spam_mail_data.csv` | 5,572 | 2 | Spam/ham text classification |
| `house_price_data.csv` | 1,460 | 81 | House price regression |
| `loan_status_data.csv` | 614 | 13 | Loan approval classification |

## Mini-Projects

- **Rock vs Mine Prediction**: Logistic Regression classification on sonar signal features.
- **Diabetes Prediction**: SVM and Logistic Regression comparison with feature scaling.
- **Spam Mail Prediction**: Text cleaning, TF-IDF vectorization, and Logistic Regression.
- **House Price Prediction**: Regression workflow with scaling, residual analysis, and feature coefficients.
- **Loan Status Prediction**: Missing-value handling, encoding, scaling, and SVM kernel comparison.

## Skills Practiced

- Python programming for data analysis
- NumPy vectorization and array operations
- pandas data cleaning and exploration
- Data visualization with matplotlib and seaborn
- Missing-value handling, scaling, encoding, and stratified splitting
- Classification and regression workflows with scikit-learn
- Model evaluation using confusion matrices, precision, recall, F1, ROC curves, MSE, MAE, and R2
- Cross-validation and basic hyperparameter search
- Implementing Linear Regression and Logistic Regression from scratch
- Understanding overfitting, underfitting, regularization effects, and bias-variance tradeoff

## How to Run

1. Clone or download this repository.
2. Open the folder in VS Code, JupyterLab, or another notebook environment.
3. Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open any notebook from the weekly folders and run the cells in order.

Most notebooks either use synthetic data created inside the notebook or read CSV files from the local `dataset/` folder.

## Reports

The `Reports/` directory contains:

- `SOS_POA_24B2289.pdf` - Plan of action
- `Midterm_Report_SOS_24B2289.pdf` - Midterm progress report

## Author

**Mohit Khyalia**  
Summer of Science 2026  
IIT Bombay
