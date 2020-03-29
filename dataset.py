import os

import pandas as pd

ID_COLUMN_NAME = 'PassengerId'
LABEL_COLUMN_NAME = 'Survived'

_project_path = os.path.join(os.environ['HOME'], 'kaggle', 'titanic')
_data_dir = os.path.join(_project_path, 'data')
_dataset_path = os.path.join(_data_dir, 'train.csv')
_testset_path = os.path.join(_data_dir, 'test.csv')


def load_test_set():
    X_test = pd.read_csv(_testset_path).set_index(ID_COLUMN_NAME)
    return X_test


def load_training_set():
    dataset_df = pd.read_csv(_dataset_path).set_index(ID_COLUMN_NAME)
    attributes_columns = [
        col_name
        for col_name in dataset_df.columns
        if col_name != LABEL_COLUMN_NAME
    ]
    X_train = dataset_df[attributes_columns]
    y_train = dataset_df[LABEL_COLUMN_NAME]
    return X_train, y_train
