from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE: Final[int] = 42

raw_df = pd.read_csv('bank-customer-churn-prediction-dlu/train.csv')

# Split data into train and validation
_train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=RANDOM_STATE, stratify=raw_df['Exited'])
train_df, val_df = train_test_split(_train_val_df, test_size=0.25, random_state=RANDOM_STATE, stratify=_train_val_df['Exited'])

input_cols = train_df.columns[1:-1]  # to drop 'id' and 'Exited' columns
target_col: Final[str] = 'Exited'

# Prepare input data for the model
train_inputs = train_df[input_cols]
train_targets = train_df[target_col]

val_inputs = val_df[input_cols]
val_targets = val_df[target_col]

test_inputs, test_targets = test_df[input_cols], test_df[target_col]

# Define numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

# Define preprocessing for numeric features (normalize them)
# Impute missing values with the median does not make sense, because there are no missing values in numeric columns
# So, we will only scale the numeric columns
numeric_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    # NOTE: OR we can try to use MinMaxScaler here since 'Balance' and 'EstimatedSalary' columns have large range of values
])

# Define preprocessing for categorical features (encode them)
# Impute missing values with the constant 'missing' does not make sense, because there are no missing values in categorical columns
# So, we will only encode the categorical columns
categorical_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)