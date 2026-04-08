import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, explained_variance_score
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import matplotlib.pyplot as plt
import seaborn as sns


target_idx=19
feat_select=0
factor_analysis=1
rand_st=1 # for reproducibility


ds = pd.read_csv('Tech_Use_Stress_Wellness.csv')

target = "weekly_depression_score"
assert target in ds.columns, f"Variable of interest '{target}' not found."

# To prevent data leakage, the user_id and any columns that are directly related
# to the target variable are removed
ds.drop(['user_id', 'mental_health_score', 'weekly_anxiety_score'], axis=1, inplace=True)

# Convert boolean columns to integers
boolean_cols = ['uses_wellness_apps', 'eats_healthy']
ds[boolean_cols] = ds[boolean_cols].astype(int)

# Map gender to binary values
ds["gender"] =  ds["gender"].map({"Male" : 1, "Female" : 0})
print(ds["gender"])
# One-hot encode the location_type column
ds = pd.get_dummies(ds, columns=["location_type"], drop_first=True)

# Check for missing values and handle them (e.g., by dropping)
missing_values = ds.isnull().sum()
print("Missing values:\n", missing_values)
ds.dropna(inplace=True)


predictors=ds.drop(columns=ds["weekly_depression_score"])
target=ds["weekly_depression_score"]

np_predictors = np.array(predictors)
np_target = np.array(target)

data_train, data_test, target_train, target_test = train_test_split(np_predictors, np_target, test_size=0.35, random_state=rand_st)

# count number of males and females
gender_counts = ds['gender'].value_counts()
print("Gender Counts:\n", gender_counts) 

# Exploratory Data Analysis (EDA)

print(target.describe())
print("Skewness:", target.skew())

# Create a visual to show the distribution of the gender variable
sns.countplot(x='gender', data=ds)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=[' Female   ', 'Male   '])                      
plt.show()

# Create heatmap of highly correlated features and target variable



correlation_matrix = ds.corr()
highly_correlated_features = correlation_matrix[target_column][abs(correlation_matrix[target_column]) > 0.5].index.tolist()
if target_column not in highly_correlated_features:
    highly_correlated_features.append(target_column)
plt.figure(figsize=(10, 8))
sns.heatmap(ds[highly_correlated_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Highly Correlated Features with Target Variable')
plt.tight_layout()
plt.show()


  


sns.boxplot(x='gender', y=target_column, data=ds)
plt.title('Boxplot of Depression Scores by Gender')
plt.xlabel('Gender')
plt.ylabel('Weekly Depression Scores')
plt.xticks(ticks=[0, 1], labels=[' Female', 'Male'])
plt.show()


# Visualize the distribution of the target variable
sns.histplot(target, kde=True)
plt.title('Distribution of Depression Scores')
plt.xlabel("Weekly Depression Scores")
plt.ylabel('Frequency')
plt.show()


# Calculate correlation matrix
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()  



screen_cols = ['daily_screen_time_hours','phone_usage_hours','laptop_usage_hours','tablet_usage_hours','tv_usage_hours']
sns.pairplot(ds[screen_cols])
plt.suptitle('Pairplot of Screen Time Variables', y=1.02)
plt.show()




sns.barplot(x='uses_wellness_apps', y='weekly_depression_score', data=ds)
plt.title('Average Depression Scores by Wellness App Usage')
plt.xlabel('Wellness App Usage')
plt.ylabel('Average Weekly Depression Scores')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()
sns.barplot(x='eats_healthy', y='weekly_depression_score', data=ds)
plt.title('Average Depression Scores by Healthy Eating Habits')
plt.xlabel('Healthy Eating')
plt.ylabel('Average Weekly Depression Scores')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()  



