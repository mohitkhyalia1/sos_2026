# Week 06 — Deep Learning Infrastructure

Mohit Khyalia | Summer of Science 2026 | IIT Bombay

---

Week 6 builds the two foundational deliverables that Week 8 depends on: neural network understanding (from scratch through Keras) and the complete Spaceship Titanic preprocessing pipeline.

## Notebooks

- **neural_network_foundations.ipynb** — Single neuron as Logistic Regression, sigmoid/ReLU/tanh activations and vanishing gradients, XOR failure with one layer, two-layer network forward + backward pass in NumPy. No external dataset — pure mathematical demonstration connecting to `logistic_regression_from_scratch.ipynb` (Week 3).

- **mlp_with_sklearn.ipynb** — MLPClassifier architecture and activation comparison, alpha as L2 regularization (connects to `regularization_and_feature_selection.ipynb`), GridSearchCV over architecture and alpha inside a Pipeline. Applied to the loan status dataset. The model-in-pipeline pattern carries directly into Week 7.

- **deep_learning_keras.ipynb** — **(Merged notebook)** Part A: Keras Sequential API, Dropout, EarlyStopping applied to tabular binary classification. Part B: systematic training strategy comparisons — SGD vs Adam, learning rate effects, batch size, dropout rates. Establishes the recommended configuration (Adam lr=0.001, dropout=0.3, batch=32, EarlyStopping patience=20) applied in Week 8 without modification.

- **spaceship_titanic_preprocessing.ipynb** — Implements all Feature Engineering Decisions from `spaceship_titanic_eda.ipynb`: Cabin parsing, spending aggregation, IsSpender flag, AgeGroup bins, ColumnTransformer with log1p for skewed columns, one-hot encoding. Fits the preprocessor on the training fold only. Produces `X_tr_t`, `X_val_t`, `X_test_t`, `feature_names` — the direct inputs to all Week 7 and Week 8 modelling.

## Key Takeaways

- A single neuron with sigmoid is Logistic Regression — neural networks generalise this by stacking layers.
- ReLU avoids vanishing gradients; its gradient is either 0 or 1 rather than shrinking toward zero.
- Adam converges faster than SGD on tabular data; lr=0.001 is the reliable default.
- EarlyStopping removes the need to manually choose epoch count — it is used in every deep learning notebook.

---
*Week 06 — Mohit Khyalia, Summer of Science 2026, IIT Bombay*
