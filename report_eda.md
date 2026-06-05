# Exploratory Data Analysis Report
## Dataset Source:
The dataset used for this project is the Pima Indians Diabetes Dataset. It contains medical diagnostic measurements collected from female patients of Pima Indian heritage. The objective is to predict whether a patient has diabetes based on several health-related features.

## Dataset size:
Rows: 768
Columns: 9
Target Variable: Outcome

## Class Balance:
The target variable Outcome contains two classes:
0 → Non-Diabetic
1 → Diabetic

## Class distribution:
Non-Diabetic: 500 samples (65.1%)
Diabetic: 268 samples (34.9%)
The dataset is moderately imbalanced, with more non-diabetic cases than diabetic cases.

## Missing Value Strategy:
The dataset does not contain explicit missing values (NaN).However, some features contain zero values that are physiologically impossible and may represent missing information.

Examples: 
Glucose
BloodPressure
SkinThickness
Insulin
BMI

For initial analysis, the dataset was kept unchanged.
Future preprocessing may include:
1.Replacing invalid zeros with NaN
2.Median imputation
3.Mean imputation

## Feature Distributions
Histograms were generated for all eight input features:
1.Pregnancies
Right-skewed distribution.
Most patients had fewer pregnancies.
2.Glucose
Approximately bell-shaped.
Higher glucose levels are associated with diabetes.
3.BloodPressure
Concentrated around normal blood pressure values.
4.SkinThickness
Contains many low and zero values.
5.Insulin
Highly skewed with several extreme values.
6.BMI
Most observations lie between 25 and 40.
7.DiabetesPedigreeFunction
Strong right-skewed distribution.
8.Age
Majority of patients are between 20 and 40 years old.

## Key Insights:
Insight 1:
Glucose shows a wide range of values and is likely to be one of the most influential features for predicting diabetes.

Insight 2:
Insulin and SkinThickness contain a significant number of zero values, suggesting potential missing-data problems that should be addressed during preprocessing.

Insight 3:
The dataset contains approximately 35% diabetic cases and 65% non-diabetic cases, indicating moderate class imbalance that may affect model performance.

## Conclusion:
The exploratory data analysis provided an understanding of the dataset structure, class distribution, and feature behavior. The identified data quality issues and class imbalance considerations will guide the preprocessing and model development stages.