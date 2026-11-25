# Psychometric Drug Consumption Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Fast%20GBDT-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model%20Evaluation-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance machine learning pipeline for multi-target classification of substance consumption risk based on psychometric and personality traits (NEO-FFI-R, BIS-11, and ImpSS).

---

## 🧠 Project Architecture & Methodology

Substance use prediction poses severe class imbalance and subtle psychological correlation challenges. This solution leverages personality profiling metrics combined with gradient boosting and rigorous Out-Of-Fold (OOF) cross-validation.

### Target Classes
Multi-target prediction across 5 distinct substances:
- `Marijuana`
- `LSD`
- `Mushroom`
- `Psychotropic`
- `Ecstasy (Ex)`

---

## 🔬 Key Engineering Highlights

### 1. Domain-Specific Psychometric Feature Engineering
Engineered composite psychometric traits representing novelty-seeking vs. self-control interactions:
$$\text{Psychedelic Affinity} = 2.5 \cdot \text{OpennessScore} + \text{NoveltySeekingScore} - \text{SelfDisciplineScore}$$

### 2. Leakage-Free Stratified K-Fold CV
- Applied `StratifiedKFold` across target classes to preserve positive class distribution across folds.
- Stored full Out-Of-Fold (OOF) probability distributions before generating test inferences to prevent optimistic cross-validation bias.

### 3. Model Optimization
- Gradient Boosting using `LGBMClassifier` tuned for discrete decision boundaries.
- Evaluation metric: **Macro F1-Score** to penalize majority-class bias on imbalanced classes.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PooyaCheshmehkhavar/Drug-Classification-ML.git
   cd Drug-Classification-ML
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn lightgbm joblib jupyter
   ```

3. **Run the Notebook:**
   ```bash
   jupyter notebook Drug.ipynb
   ```

---

## 👤 Author
**Pooya Cheshmehkhavar**  
- GitHub: [@PooyaCheshmehkhavar](https://github.com/PooyaCheshmehkhavar)
