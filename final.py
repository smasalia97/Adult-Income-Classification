# Author - Saiyam Maunik Masalia
# Assignment - Final Implementation [AI539 Machine Learning Challenges Winter 2024]
# Date - 03/18/2024


import pandas as pd
import time
from scipy.stats.mstats import winsorize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="In version 1.5 onwards, subsample=200_000 will be used by default.*",
)

s = time.time()


def load_data():
    file_path = "adult.csv"  # Make sure both files are in same directory
    return pd.read_csv(file_path, na_values=["?"]).drop_duplicates()


def preprocess_data(df):
    # Convert all columns to string type
    df = df.astype(str)

    # Encode categorical data
    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])
    return df


def train_and_evaluate(classifiers, X_train, X_test, y_train, y_test):
    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred)
        results.append(
            {
                "Accuracy (%)": round(accuracy, 3),
                "F1 Score": f1,
            }
        )
    return pd.DataFrame(results)

# Load data
data = load_data()

data = preprocess_data(data)

# Split data into features and target variable
X = data.drop("income", axis=1)
y = data["income"]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize classifiers
classifiers = {
    # "RF": RandomForestClassifier(random_state=0),
    "XGB": XGBClassifier(
        random_state=42,
    ),
}

# Train and evaluate classifiers
results = train_and_evaluate(classifiers, X_train, X_test, y_train, y_test)
# print("Results for Baseline:")
# print(results, "\n")
# print(
#     "------------------------------------------------------------------------------------------------"
# )


# Apply winsorization to training data
def winsorize_dataframe(df, limits):
    df_winsorized = df.copy()
    for column in df.columns:
        df_winsorized[column] = winsorize(df[column], limits=limits)
    return df_winsorized


X_train_winsor = winsorize_dataframe(X_train, limits=[0.05, 0.05])

# Train and evaluate classifiers after winsorization
results_winsor = train_and_evaluate(
    classifiers, X_train_winsor, X_test, y_train, y_test
)

# print("Results after Winsorizing:")
# print(results_winsor, "\n")
# print(
#     "------------------------------------------------------------------------------------------------"
# )


# Apply imputation with mean
def impute_outliers(df, method):
    df_imputed = df.copy()
    for column in df.columns:
        lower_limit = df_imputed[column].quantile(0.05)
        upper_limit = df_imputed[column].quantile(0.95)
        outlier_mask = (df_imputed[column] < lower_limit) | (
            df_imputed[column] > upper_limit
        )
        mean_value = df_imputed[column].mean()
        df_imputed.loc[outlier_mask, column] = mean_value.astype(df[column].dtype)
    return df_imputed


X_train_mean_imputed = impute_outliers(X_train, method="mean")

# Train and evaluate classifiers after imputation
results_mean_imputed = train_and_evaluate(
    classifiers,
    X_train_mean_imputed,
    X_test,
    y_train,
    y_test,
)

# Apply binning
est = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
X_train_binned = est.fit_transform(X_train)
X_test_binned = est.transform(X_test)

# Train and evaluate classifiers on binned data
results_binned = train_and_evaluate(
    classifiers, X_train_binned, X_test_binned, y_train, y_test
)
# print("Results after Binning", "\n")
# print(results_binned, "\n")
# print(
#     "------------------------------------------------------------------------------------------------"
# )

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Drop rows with missing values
data = load_data()
data_NA = data.dropna()
data_NA_pre = preprocess_data(data_NA)

# Split data into features and target variable after dropping rows with missing values
X_dropna = data_NA_pre.drop("income", axis=1)
y_dropna = data_NA_pre["income"]

# Split the dataset into train and test sets again
X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna = train_test_split(
    X_dropna, y_dropna, test_size=0.2, random_state=42
)

# Train and evaluate classifiers after dropping rows with missing values
results_dropna = train_and_evaluate(
    classifiers,
    X_train_dropna,
    X_test_dropna,
    y_train_dropna,
    y_test_dropna,
)
# print("Results after Dropping Rows with Missing Values:")
# print(results_dropna, "\n")
# print(    "------------------------------------------------------------------------------------------------")

data_MI = load_data()
data_MI_NA = data_MI.fillna(data_NA_pre.mean())

# Preprocess the data after imputation with mean
data_MI_NA = preprocess_data(data_MI_NA)

X_MI = data_MI_NA.drop("income", axis=1)
y_MI = data_MI_NA["income"]

# Split dataset into train and test sets
X_train_MI, X_test_MI, y_train_MI, y_test_MI = train_test_split(
    X_MI, y_MI, test_size=0.2, random_state=42
)

# Train and evaluate classifiers after imputation with mean
results_mean_NA = train_and_evaluate(
    classifiers,
    X_train_MI,
    X_test_MI,
    y_train_MI,
    y_test_MI,
)
# print("Results after Imputation with Mean:")
# print(results_mean_NA, "\n")

# Interpolation: Linear Interpolation
data_inter = load_data()

X_inter = data_inter.drop("income", axis=1)
y_inter = data_inter["income"]

# Convert object columns to numeric type
X_inter_numeric = X_inter.apply(pd.to_numeric, errors="coerce")

X_train_inter, X_test_inter, y_train_inter, y_test_inter = train_test_split(
    X_inter_numeric, y_inter, test_size=0.2, random_state=42
)

# Perform linear interpolation on the data
X_train_linear_interpolate = X_train_inter.interpolate(method="linear")
X_test_linear_interpolate = X_test_inter.interpolate(method="linear")

# Perform one-hot encoding on categorical variables
X_train_encoded = pd.get_dummies(X_train_linear_interpolate)
X_test_encoded = pd.get_dummies(X_test_linear_interpolate)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable
y_train_encoded = label_encoder.fit_transform(y_train_inter)
y_test_encoded = label_encoder.transform(y_test_inter)

# Train and evaluate classifiers on data with one-hot encoding
results_linear_interpolate_encoded = train_and_evaluate(
    classifiers,
    X_train_encoded,
    X_test_encoded,
    y_train_encoded,
    y_test_encoded,
)

# print("Results after Linear Interpolation:")
# print(results_linear_interpolate, "\n")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Define resampling techniques
over_sampler = RandomOverSampler(random_state=42)
smote = SMOTE(random_state=42)

# Apply resampling techniques to training data
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train and evaluate classifiers after resampling

results_over = train_and_evaluate(
    classifiers,
    X_train_over,
    X_test,
    y_train_over,
    y_test,
)

results_smote = train_and_evaluate(
    classifiers, X_train_smote, X_test, y_train_smote, y_test
)


# print("Results after Over Sampling:")
# print(results_over, "\n")
# print("Results after SMOTE:")
# print(results_smote, "\n")

# Initialize classifiers with class weights
classifiers_weighted = {
    "RF (Class Weight)": RandomForestClassifier(class_weight="balanced", random_state=0)
}

# Train and evaluate classifiers with class weights
results_weighted = train_and_evaluate(
    classifiers_weighted,
    X_train,
    X_test,
    y_train,
    y_test,
)

# # Print results
# print("Results with Class Weights:")
# print(results_weighted, "\n")


# ---------------------------------------------------------------- Results.csv \/ ----------------------------------------------------------------
# Concatenate all results DataFrames
all_results = pd.concat(
    [
        results.assign(Preprocessing="Base"),
        results_winsor.assign(Preprocessing="Winsorization 5/95"),
        results_mean_imputed.assign(Preprocessing="Mean Imputation in Outliers"),
        results_binned.assign(Preprocessing="Binning in Outliers"),
        results_dropna.assign(Preprocessing="Dropping NA"),
        results_mean_NA.assign(Preprocessing="Mean Imputation in NA"),
        results_linear_interpolate_encoded.assign(Preprocessing="Linear Interpolation"),
        results_over.assign(Preprocessing="Over Sampling"),
        results_smote.assign(Preprocessing="SMOTE"),
        results_weighted.assign(Preprocessing="Class Weights"),
    ],
    ignore_index=True,
)

# Reorder columns to make "Preprocessing" the first column
all_results = all_results[["Preprocessing", "Accuracy (%)", "F1 Score"]]

e = time.time()

# Export all results to a single CSV file
all_results.to_csv("all_results.csv", index=False)
print("9 Strategies to handle 3 challenges usin XGBoost:\n\n", all_results)

# Find the row with maximum accuracy
max_accuracy_row = all_results.loc[all_results["Accuracy (%)"].idxmax()]

# Print the row with maximum accuracy
print("\nMAX Accuracy:", max_accuracy_row)
print("\nTotal time taken:", round(e - s, 3), "seconds")
