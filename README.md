# **Depression Score Prediction using Lifestyle & Behavioral Factors**

Machine Learning (ML) pipeline that predicts weekly depression scores using behavioral and lifestyle factors. Four different regression models are compared 
across three feature selection techniques, with XAI for clinical interpretability. 

## **Overview** 

This project seeks to determine if behavioral and lifestyle data can be used to effectively predict depression severity scores. The additional focus of this
project is to investigate different methods of feature selection -- specifically, the **INTERACT** algorithm is implemented from scratch and is compared against
two standard feature selection strategies -- SelectKBest with Mutual Information (MI) and Recursive Feature Elimination (RFE). 

All of the models are evaluated based on their average RMSE and explained variance using 5-fold cross-validation. 


### **Models**
| Model | Description |
|---|---|
| Random Forest (RF) | Ensemble of decision trees |
| Gradient Boosting (GB) | Sequential boosting ensemble |
| SVR | Support Vector Regression (linear kernel) |
| MLP | Multi-layer Perceptron | 


### **Feature Selection Methods**
| Method | Description |
|---|---|
| None | All features used |
| INTERACT | Custom implementation -- filters out redundant features using symmetrical uncertainty | 
| SelectKBest (Mutual Information) | Selects top-k features by mutual information with target | 
| RFE | Recursive Feature Elimination using RF as base estimator | 



## **Results**
INTERACT feature selection consistently outperformed other models in cross-validation performanc, with the lowest CV RMSE and highest CV 
Explained Variance

| Model | Feature Selection | RMSE | Explained Variance | CV Expl. Var. Mean | CV RMSE Mean |
|---|---|---|---|---|---|
| RF | None | 2.7933 | 0.6462 | 0.6472 | 2.7744 |
| RF | INTERACT | **2.7520** | **0.6567** | **0.6672** | **2.6946** |
| RF | MI | 2.8043 | 0.6435 | 0.6444 | 2.7853 |
| RF | RFE | 2.7968 | 0.6454 | 0.6451 | 2.7827 |
| GB | None | 2.8212 | 0.6393 | 0.6427 | 2.7930 |
| GB | INTERACT | **2.7519** | **0.6567** | **0.6672** | **2.6946** |
| GB | MI | 2.8022 | 0.6441 | 0.6465 | 2.7778 |
| GB | RFE | 2.8272 | 0.6377 | 0.6417 | 2.7971 |
| SVR | None | 2.7684 | 0.6528 | 0.6620 | 2.7157 |
| SVR | INTERACT | 2.7565 | 0.6567 | **0.6672** | 2.6956 |
| SVR | MI | 2.7567 | 0.6560 | 0.6652 | 2.7031 |
| SVR | RFE | 2.7589 | 0.6554 | 0.6653 | 2.7025 |
| MLP | None | 2.8601 | 0.6293 | 0.6414 | 2.7960 |
| MLP | INTERACT | 2.7660 | 0.6539 | 0.6661 | 2.6987 |
| MLP | MI | 2.7866 | 0.6482 | 0.6593 | 2.7253 |
| MLP | RFE | 2.7827 | 0.6490 | 0.6554 | 2.7411 |

### Implementation of INTERACT
INTERACT is a filter-based feature selection algorithm that uses symmetrical uncertainty (SU) -- a form of mutual information -- to rank features. 
It removes features based on their correlation with other features rather than with the target variable. 

The INTERACT implementation (INTERACTSelector) allows the algorithm to work as a sklearn-compatible BaseEtimator / TransformerMixin, making it drop-in compatible 
with sklearn-Pipeline.

### **Key Components**
* entropy() -- calculates the Shannon entropy of a discrete variable
* mutual_info() -- entropy-based mutual information
* symmetrical_uncertainty() -- normalized MI between two variables
* interact() -- core selection loop for algorithm
* INTERACTSelector -- sklearn transformer wrapper

## **Dataset**
Tech Use, Stress & Wellness Dataset -- sourced from [Kaggle](https://www.kaggle.com/datasets/nagpalprabhavalkar/tech-use-and-stress-wellness). 

* 5,000 samples
* Behavioral and lifestyle features (screen time, stress, sleep, diet, etc.)
* Target variable: weekly_depression_score


## **Usage**

**1. Install dependencies**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap
```

**2. Download the dataset**

Download `Tech_Use_Stress_Wellness.csv` from [Kaggle](https://www.kaggle.com/) and place it in the project root directory.

**3. Run the pipeline**
```bash
python model.py
```

**4. Select a feature selection method when prompted**
```
0 - No feature selection
1 - SelectKBest (Mutual Information)
2 - INTERACT (custom implementation)
3 - Recursive Feature Elimination (RFE)
```

## **Dependencies**

Python 3.8+
scikit-learn
numpy
pandas
matplotlib
seaborn
shap

## **Paper**
*Coming soon* -- an accompanying paper detailing the methodology, results, and clinical implication

## **Author**
**Peter Teresi** [GitHub](https://github.com/pteresi05) - [LinkedIn](https://linkedin.com/in/pete-teresi/)


