import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def main():
    """
    Script to split the dataset into training and testing sets.
    The dataset is read from a CSV file, and the features and target variable are separated.
    The data is then split into training and testing sets, which are saved as pickle files.
    """

    # Change working directory to current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data = pd.read_csv('../../data/Loan_Default.csv')
    logging.info("Data read")

    logging.info("Data Sample:\n%s", data.head())

    X = data.drop("Status", axis=1)
    y = data[["Status"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    logging.info("Data split into train and test sets")

    logging.info("X_train Sample:\n%s", X_train.head())
    logging.info("y_train Sample:\n%s", y_train.head())
    logging.info("X_test Sample:\n%s", X_test.head())
    logging.info("y_test Sample:\n%s", y_test.head())

    # Save the train and test sets to pickle files
    X_train.to_pickle('../../data/X_train.pkl')
    y_train.to_pickle('../../data/y_train.pkl')
    X_test.to_pickle('../../data/X_test.pkl')
    y_test.to_pickle('../../data/y_test.pkl')
    logging.info("Train and test sets saved to pickle files")


if __name__ == '__main__':
    main()
