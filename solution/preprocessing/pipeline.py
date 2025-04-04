import os
import pickle
import logging
import pandas as pd
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42


numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline_onehot = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
])

categorical_pipeline_ordinal = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('scaler', StandardScaler(with_mean=False))
])


def main():
    '''
    Data pipeline:
    
    '''

    # Change working directory to current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Read the data from pickle files
    X_train = pd.read_pickle('../../data/X_train.pkl')
    y_train = pd.read_pickle('../../data/y_train.pkl')
    X_test = pd.read_pickle('../../data/X_test.pkl')
    y_test = pd.read_pickle('../../data/y_test.pkl')

    # Drop columns that has information leakage and contains many missing values (rate_of_interest, Interest_rate_spread, Upfront_charges) 
    # and not needed for the model (ID, year)
    X_train = X_train.drop(columns=['ID', 'year', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges'])
    X_test = X_test.drop(columns=['ID', 'year', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges'])

    logging.info("X_train Sample:\n%s", X_train.head())
    logging.info("y_train Sample:\n%s", y_train.head())
    logging.info("X_test Sample:\n%s", X_test.head())
    logging.info("y_test Sample:\n%s", y_test.head())

    # Define the categorical and numerical features
    numerical_columns = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
    categorical_columns_onehot = [col for col in X_train.columns if X_train[col].dtype == 'object' and X_train[col].nunique() <= 5]
    categorical_columns_ordinal = [col for col in X_train.columns if X_train[col].dtype == 'object' and X_train[col].nunique() > 5]

    pipeline = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_columns),
        ('cat_onehot', categorical_pipeline_onehot, categorical_columns_onehot),
        ('cat_ordinal', categorical_pipeline_ordinal, categorical_columns_ordinal)
    ])
    logging.info("Pipeline defined")

    pipeline.fit(X_train, y_train)
    logging.info("Pipeline fitted")

    # Save the pipeline to a pickle file
    with open('../models/pipelines/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    X_train = pipeline.transform(X_train)
    X_test = pipeline.transform(X_test)
    logging.info("Data transformed")

    # Convert the transformed data back to DataFrame
    # Get the feature names from the transformers
    feature_names = numerical_columns + \
                    pipeline.named_transformers_['cat_onehot'].named_steps['onehot'].get_feature_names_out(categorical_columns_onehot).tolist() + \
                    pipeline.named_transformers_['cat_ordinal'].named_steps['ordinal'].get_feature_names_out(categorical_columns_ordinal).tolist()
    # Create DataFrames with the transformed data
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    logging.info("Transformed X_train Sample:\n%s", X_train.head())
    logging.info("Transformed X_test Sample:\n%s", X_test.head())

    # Save the transformed data to pickle files
    X_train.to_pickle('../processed_data/X_train_transformed.pkl')
    y_train.to_pickle('../processed_data/y_train_transformed.pkl')
    X_test.to_pickle('../processed_data/X_test_transformed.pkl')
    y_test.to_pickle('../processed_data/y_test_transformed.pkl')
    logging.info("Transformed data saved to pickle files")


if __name__ == '__main__':
    main()
