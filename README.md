# Heart Failure

![GitHub repo size](https://img.shields.io/github/repo-size/rm-kara/Heart-Failure-Predictions)
![GitHub stars](https://img.shields.io/github/stars/rm-kara/Heart-Failure-Predictions)
![GitHub contributors](https://img.shields.io/github/contributors/rm-kara/Heart-Failure-Predictions)
![GitHub forks](https://img.shields.io/github/forks/rm-kara/Heart-Failure-Predictions)

## Table of contents
* [Project Overview](#project-overview)
* [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installing Requirements](#installing-requirements)
* [EDA Highlights](#eda-highlights)
* [Tested Models](#tested-models)
* [Best Performance](#best-performance)
* [Resources](#resources)

## Project Overview
This data set contains 12 features related to whether a patient will suffer from **heart failure**.
It's crucial to be aware that there are two target columns: **time** and **DEATH_EVENT**  
**DEATH_EVENT** describes whether a patient died or was censored. Censoring means the scientist lost contact with the patient.  
The **time** column indicates when the patient died or was censored.  
Since we don't have the information when a patient will die or get censored we cannot use **time** as a feature.

The data was cleaned using **Pandas** and **Numpy**, visualizations were developed with **Seaborn** and **Matplotlib**.  
Transformational steps such as encoding categorical- and numerical variables were implemented by using Scikit-learn's **Pipeline, StandardScaler, OneHotEncoder and ColumnTransformer** modules.  
Imblearn's **SMOTE (synthetic minority oversampling technique)** was used to solve the imbalanced data distribution.  
Different models were then compared and their performance evaluated using **Stratified K-Fold cross-validation**. Finally, the best model was selected and optimized using **GridSearchCV**.

## Getting Started

### Prerequisites
**Python Version:** 3.7  
**Packages:**
* pandas 1.1.0 
* numpy 1.19.1
* scikit-learn 0.23.2
* matplotlib 3.3.1
* seaborn 0.10.1
* xgboost 1.2.0

### Installing Requirements
To create a new anaconda environment, download [conda_requirements.txt](https://github.com/rm-kara/Heart-Failure-Predictions/tree/master/requirements/conda_requirements.txt) and enter the following command:  
```
<conda create --name <env> --file conda_requirements.txt>
```
To install the packages with pip, download [requirements.txt](https://github.com/rm-kara/Heart-Failure-Predictions/tree/master/requirements/requirements.txt) and enter the following command:  
```
<pip install -r requirements.txt>
```
## EDA Highlights
**Age distribution by gender:** 
![alt text](https://github.com/rm-kara/Heart-Failure-Predictions/blob/master/img/Age-Dist-Sex.png "Age Distribution Gender")
***
**Deaths by gender and age:**
![alt text](https://github.com/rm-kara/Heart-Failure-Predictions/blob/master/img/Death-Dist-Sex.png "Death by Age and Gender")
***
**Different factors and their influence on age:**
![alt text](https://github.com/rm-kara/Heart-Failure-Predictions/blob/master/img/Age-Influencer%3F.png "Factors on Age")
***
**Death frequency and elapsed time:**
![alt text](https://github.com/rm-kara/Heart-Failure-Predictions/blob/master/img/Death-Details.png "Deaths Frequency")

## Tested Models
* XGBClassifier: 0.718
* Gradient Boosting: 0.733
* KNeighbors: 0.681
* SVM: 0.735
* Logistic Regression: 0.722 

## Best Performance
* **Tuned SVM Accuracy: 0.759**  
### Confusion Matrix
![alt text](https://github.com/rm-kara/Heart-Failure-Predictions/blob/master/img/Confusion%20Matrix.png "Confusion Matrix")

## Resources
* Link to Data: 
    - [Kaggle Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)
* GridSearchCV: 
    - [Splitting Data](https://stackoverflow.com/questions/45394527/do-i-need-to-split-data-when-using-gridsearchcv)
* SMOTE: 
    - [SMOTE With GridSearch](https://stackoverflow.com/questions/50245684/using-smote-with-gridsearchcv-in-scikit-learn)
    - [Standardization With SMOTE](https://stats.stackexchange.com/questions/363312/normalization-standardization-should-one-do-this-before-oversampling-undersampl)
* Hyperparameters GradientBoosting:
    - [Tune Hyperparameters](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/)
    - [More on tuning Hyperparameters](https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624)
