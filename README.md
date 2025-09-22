[![Repo](https://img.shields.io/badge/GitHub-Explainable--Machine--Learning--Loan--Prosecution-blue?logo=github)](https://github.com/U1186204/Explainable-Machine-Learning-Loan-Prosecution)
[![CI Pipeline](https://github.com/U1186204/Explainable-Machine-Learning-Loan-Prosecution/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/U1186204/Explainable-Machine-Learning-Loan-Prosecution/actions/workflows/ci-pipeline.yml)
[![License](https://img.shields.io/github/license/U1186204/Explainable-Machine-Learning-Loan-Prosecution)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/U1186204/Explainable-Machine-Learning-Loan-Prosecution/blob/main/loan_prosecution.ipynb)

# Explainable-Machine-Learning-Loan-Prosecution

### Description
This exercise evaluates a loan denial using Explainable ML models. We employ SHAP and LIME, which lead to the conclusion that the model strongly relies on discriminatory variables when underwriting loans to an individual.

---
## Project Structure
```
.
├── LICENSE
├── README.md
├── llm_log.txt
├── loan_prosecution.ipynb
└── requirements.txt
```

---
## XAI Methods Used
This project uses two key XAI (Explainable AI) techniques to interpret the "black box" decisions of a Random Forest model trained on the UCI Adult Income dataset.

### SHAP (SHapley Additive exPlanations)
SHAP uses a game theory approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations, assigning each feature an importance value—a "Shapley value"—for a particular prediction.

* **In this project:** We used a SHAP force plot to analyze the specific denial for "Jane Dow." It provided direct evidence that discriminatory factors like **sex** and irrational factors like her **high education level** and **long work hours** were negatively impacting the model's decision. A global summary plot also confirmed that demographic features like `relationship` and `age` were major drivers for the model overall.

### LIME (Local Interpretable Model-agnostic Explanations)
LIME explains the predictions of any classifier by learning an interpretable model (like a linear model) locally around the prediction. It answers the question, "Why did the model make this specific decision?" by showing which features supported or contradicted the outcome.

* **In this project:** LIME provided a clear, weighted list of factors contributing to Jane's denial. It corroborated the SHAP findings by highlighting that her **marital status** and **relationship status** were used as justifications for the denial, reinforcing the argument that the model relies on discriminatory biases rather than purely financial indicators.
