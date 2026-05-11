# =========================================================
# HOUSE PRICE PREDICTION PROJECT
# Data Analysis using Python
# =========================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# READ DATASET
# -----------------------------

# For Excel file
df = pd.read_excel("HousePrice.xlsx")

# Remove rows where SalePrice is missing
df = df.dropna(subset=['SalePrice'])

# -----------------------------
# DISPLAY ALL COLUMNS
# -----------------------------

print("\n================ ALL COLUMNS ================\n")
print(df.columns)

# -----------------------------
# DISPLAY FIRST 100 ROWS
# -----------------------------

print("\n================ FIRST 100 ROWS ================\n")
print(df.head(100))

# -----------------------------
# COLUMN INFORMATION
# -----------------------------

print("\n================ COLUMN INFORMATION ================\n")
print(df.info())

# -----------------------------
# DATA DESCRIPTION
# -----------------------------

print("\n================ DATA DESCRIPTION ================\n")
print(df.describe())

# =========================================================
# TASK 2 : DATA CLEANING AND ANALYSIS
# =========================================================

# -----------------------------
# Q1 : CHECK MISSING VALUES
# -----------------------------

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

# -----------------------------
# Q2 : FEATURES WITH NAN VALUES
# -----------------------------

nan_features = [feature for feature in df.columns if df[feature].isnull().sum() > 0]

print("\n================ FEATURES WITH NaN VALUES ================\n")
print(nan_features)

# -----------------------------
# Q3 : MEAN SALE PRICE FOR
# MISSING / PRESENT VALUES
# -----------------------------

print("\n================ MEAN SALE PRICE FOR MISSING/PRESENT DATA ================\n")

for feature in nan_features:

    data = df.copy()

    data[feature] = np.where(data[feature].isnull(), 1, 0)

    print("\nFeature:", feature)

    print(data.groupby(feature)['SalePrice'].mean())

# -----------------------------
# Q4 : COUNT NUMERICAL FEATURES
# -----------------------------

numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']

print("\n================ NUMBER OF NUMERICAL FEATURES ================\n")

print(len(numerical_features))

# -----------------------------
# Q5 : FIRST 5 ROWS OF
# NUMERICAL FEATURES
# -----------------------------

print("\n================ FIRST 5 ROWS OF NUMERICAL FEATURES ================\n")

print(df[numerical_features].head())

# -----------------------------
# Q6 : YEAR FEATURES VS SALEPRICE
# -----------------------------

year_features = ['YearBuilt', 'YearRemodAdd']

for feature in year_features:

    plt.figure(figsize=(8,5))

    plt.scatter(df[feature], df['SalePrice'])

    plt.xlabel(feature)

    plt.ylabel("SalePrice")

    plt.title(f"{feature} vs SalePrice")

    plt.show()

# -----------------------------
# Q7 : DISCRETE VARIABLES VS SALEPRICE
# -----------------------------

discrete_features = ['OverallCond']

for feature in discrete_features:

    data = df.groupby(feature)['SalePrice'].median()

    plt.figure(figsize=(8,5))

    data.plot.bar()

    plt.xlabel(feature)

    plt.ylabel("Median SalePrice")

    plt.title(f"{feature} vs SalePrice")

    plt.show()

# -----------------------------
# Q8 : CONTINUOUS VARIABLES VS SALEPRICE
# -----------------------------

continuous_features = ['LotArea', 'TotalBsmtSF']

for feature in continuous_features:

    plt.figure(figsize=(8,5))

    plt.scatter(df[feature], df['SalePrice'])

    plt.xlabel(feature)

    plt.ylabel("SalePrice")

    plt.title(f"{feature} vs SalePrice")

    plt.show()

# -----------------------------
# Q9 : HISTOGRAM ANALYSIS
# -----------------------------

for feature in continuous_features:

    plt.figure(figsize=(8,5))

    df[feature].hist(bins=30)

    plt.xlabel(feature)

    plt.ylabel("Frequency")

    plt.title(f"Histogram of {feature}")

    plt.show()

# -----------------------------
# Q10 : LOG TRANSFORMATION
# -----------------------------

for feature in continuous_features:

    if 0 not in df[feature].values:

        df[feature + '_log'] = np.log(df[feature])

        plt.figure(figsize=(8,5))

        df[feature + '_log'].hist(bins=30)

        plt.xlabel(feature + '_log')

        plt.ylabel("Frequency")

        plt.title(f"Log Transformed {feature}")

        plt.show()

# =========================================================
# TASK 3 : FEATURE ENGINEERING
# =========================================================

# -----------------------------
# Q1 : FIND OUTLIERS
# -----------------------------

for feature in continuous_features:

    plt.figure(figsize=(8,5))

    sns.boxplot(x=df[feature])

    plt.title(f"Boxplot of {feature}")

    plt.show()

# -----------------------------
# Q2 : CATEGORICAL FEATURES
# VS SALEPRICE
# -----------------------------

categorical_features = ['MSZoning', 'BldgType']

for feature in categorical_features:

    plt.figure(figsize=(8,5))

    data = df.groupby(feature)['SalePrice'].median()

    data.plot.bar()

    plt.xlabel(feature)

    plt.ylabel("Median SalePrice")

    plt.title(f"{feature} vs SalePrice")

    plt.show()

# -----------------------------
# Q3 : CORRELATION MATRIX
# -----------------------------

corrmat = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))

sns.heatmap(corrmat, annot=True, cmap='coolwarm')

plt.title("Correlation Matrix")

plt.show()

# -----------------------------
# Q4 : CONTINUOUS FEATURES
# VS SALEPRICE
# -----------------------------

for feature in continuous_features:

    plt.figure(figsize=(8,5))

    sns.scatterplot(x=df[feature], y=df['SalePrice'])

    plt.title(f"{feature} vs SalePrice")

    plt.show()

# -----------------------------
# Q5 : HANDLE MISSING VALUES
# -----------------------------

for feature in nan_features:

    if df[feature].dtype != 'O':

        df[feature].fillna(df[feature].median(), inplace=True)

    else:

        df[feature].fillna(df[feature].mode()[0], inplace=True)

# -----------------------------
# HANDLE CATEGORICAL VARIABLES
# -----------------------------

df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# HANDLE TEMPORAL VARIABLES
# -----------------------------

df['HouseAge'] = 2025 - df['YearBuilt']

print("\n================ FEATURE ENGINEERING COMPLETED ================\n")

# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

# -----------------------------
# INPUT AND OUTPUT
# -----------------------------

X = df.drop('SalePrice', axis=1)

y = df['SalePrice']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# LINEAR REGRESSION MODEL
# -----------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# MODEL EVALUATION
# -----------------------------

print("\n================ MODEL PERFORMANCE ================\n")

print("Mean Absolute Error (MAE):")

print(mean_absolute_error(y_test, y_pred))

print("\nR2 Score:")

print(r2_score(y_test, y_pred))

print("\n================ PROJECT COMPLETED SUCCESSFULLY ================\n")