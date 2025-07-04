## xai-l2g
# Integrating Explainable AI into Open Targets' Locus-to-Gene (L2G) Model

This repository details a project aimed at enhancing the interpretability of the Open Targets' Locus-to-Gene (L2G) model by integrating Explainable AI (XAI) techniques. The project trains an `XGBoost` model on the Open Targets gold-standard dataset and then uses SHAP and LIME to explain the model's predictions.

---

## 📖 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Objectives](#2-objectives)
- [3. Methodology](#3-methodology)
- [4. Implementation](#4-implementation)
- [5. XAI Analysis & Findings](#5-xai-analysis--findings)
- [6. Setup and Usage](#6-setup-and-usage)

---

## 1. Project Overview

The Open Targets Locus-to-Gene (L2G) model is a powerful machine learning resource for prioritizing causal genes at GWAS loci[cite: 42, 91]. However, like many complex models, its predictions are challenging to interpret. [cite_start]This "black box" nature makes it difficult for researchers to validate predictions and trust the model's reasoning, limiting its utility in clinical and research applications.

This project directly addresses this interpretability challenge by integrating state-of-the-art XAI techniques to provide meaningful and actionable explanations for the L2G model's outputs without sacrificing its predictive accuracy.

---

## 2. Objectives

The primary goals of this project are as follows:

**Train the L2G Model:** Train the L2G machine learning model using the gold-standard set of 445 curated GWAS loci, where causal genes have been identified with high confidence.
**Apply SHAP:** Use SHAP (SHapley Additive exPlanations) to identify which features (e.g., colocalization, QTL, genomic distance) have the most impact on the L2G gene prioritization score[.
**Implement LIME:** Use LIME (Local Interpretable Model-agnostic Explanations) to generate simple, local explanations for why a specific gene was prioritized for a particular disease.
**Generate Counterfactuals:** Create counterfactual explanations to show the smallest feature changes that would alter a gene's ranking, providing actionable insights for researchers.
**Develop a Dashboard:** Design and build an interactive dashboard to present the SHAP, LIME, and counterfactual results in an accessible and user-friendly format.

---

## 3. Methodology

The project follows a multi-phased methodology to systematically integrate XAI into the L2G workflow.

### Phase 1: Establishing the Predictive Model Baseline
Before explanations can be generated, a robust predictive model is established. [cite_start]The L2G model, which utilizes a gradient boosting algorithm (XGBoost), is trained on the curated **gold-standard dataset**. This initial phase ensures that the model to be explained is both accurate and founded on high-quality biological evidence.

### Phase 2: Generating Multi-Faceted Explanations
This phase focuses on applying a suite of XAI techniques to explain the model's behavior.
**Global Feature Attribution:** **SHAP** is used to compute global feature importance, identifying which genomic features have the most significant influence on gene prioritization across all predictions.
**Local Prediction Explanation:** Both **SHAP** and **LIME** are applied to generate instance-specific explanations, clarifying why a specific gene received its priority score for a given trait.
**Actionable Counterfactual Insights:** **Counterfactual Explanations** are generated to provide "what-if" scenarios, showing what minimal changes would alter a prediction.

### Phase 3: Synthesizing Explanations for User Interpretation
The final phase focuses on making the generated explanations accessible. A **Visualization and Interpretability Dashboard** is designed to present the complex outputs from the XAI analyses in an integrated and intuitive user interface.

---

## 4. Implementation

### Technical Environment
The project is implemented in **Python 3.12** using a `conda` virtual environment. Key libraries include:
* **Pandas & NumPy:** For data manipulation.
**Scikit-learn:** For machine learning utilities like `train_test_split`.
**XGBoost:** The specific library used to train the L2G model classifier.
**SHAP & LIME:** The core libraries for XAI analysis.
**Joblib:** For saving and loading the trained model.

### Data Sources & Model Training
The model integrates multiple datasets, including the **GWAS Catalog**, **UK Biobank**, and functional genomics data from **Ensembl**. For this analysis, the model was trained on the Open Targets `gwas_gold_standards.191108.tsv` dataset.

The training process involved:
1.  Selecting `association_info.neg_log_pval` and `sentinel_variant.locus_GRCh37.position` as features.
2.  Creating a binary target where a confidence of 'High' is mapped to 1.
3.  Training an `XGBClassifier` model on an 80/20 data split.
4.  The final model achieved a test accuracy of approximately **74.5%**.

---

## 5. XAI Analysis & Findings

Both SHAP and LIME identified the same two features as the most influential drivers of the model's predictions.

### Global Feature Importance
The analysis consistently shows that the model relies primarily on two features:
1.  **`sentinel_variant.locus_GRCh37.position`** (Most Important)
2.  **`association_info.neg_log_pval`**

### Local Prediction Explanation
By dissecting a single prediction using SHAP and LIME, we can see how these features interact. For the first row in the test set:
* **Positive Force (Pushes prediction higher):** A `neg_log_pval` of **13.0** increased the likelihood of a 'High' confidence prediction.
* **Negative Force (Pushes prediction lower):** A `position` of **~7.2e7** acted as a stronger counter-force, significantly decreasing the prediction probability.

### Results (Visualizations)

#### SHAP Analysis
**Global Importance (Summary Plot)**


**Local Explanation (Force Plot)**
![SHAP Force Plot](https://i.imgur.com/XqT2Jk2.png)

#### LIME Analysis
**Global Importance (Summary-Dot Plot)**


**Local Explanation (Bar Chart)**

---

## 6. Setup and Usage

To replicate this analysis, follow these steps.

### A. Prerequisites
- Python 3.9+
- Git

### B. Clone the Repository & Data
First, clone this repository. Then, clone the required dataset from Open Targets into the same root directory.
```bash
# Clone this project repository
git clone [URL_to_this_repository]

# Clone the required data repository
git clone [https://github.com/opentargets/genetics-gold-standards.git](https://github.com/opentargets/genetics-gold-standards.git)

# Navigate into the project folder
cd [repository_name]

# Create a virtual environment and install the required packages from requirements.txt.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```
## requirements
pandas
xgboost
scikit-learn
joblib
shap
lime
matplotlib
ipykernel
jupyter

Run the Notebook
Launch Jupyter and run the cells in the provided notebook (analysis.ipynb).

```

jupyter notebook
```
