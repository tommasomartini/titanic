import random

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import dataset as ds
import preprocessing as pp
import submission as sm

np.random.seed(0)
random.seed(0)


def preprocess_dataset(df):
    df = pp.replace_nan_embarked_with_unknown(df)
    df = pp.replace_nan_age_with_value(df)
    df = pp.replace_nan_fare_with_value(df)

    attributes_to_categorical = ['Sex', 'Embarked', 'Pclass']
    for attribute_name in attributes_to_categorical:
        df = pp.convert_attribute_to_categorical(df, attribute_name)

    df = pp.add_title_column(df)
    df = pp.add_ticket_number_column(df)
    df = pp.add_floor_column(df)

    df = pp.convert_categorical_columns_to_numerical(df)

    non_numerical_attributes = ['Name', 'Ticket', 'Cabin']
    df.drop(columns=non_numerical_attributes, inplace=True)

    return df


def main():
    print('Decision trees')

    X_dataset, y_dataset = ds.load_training_set()
    X_dataset = preprocess_dataset(X_dataset)
    X_train, X_val, y_train, y_val = train_test_split(X_dataset,
                                                      y_dataset,
                                                      test_size=0.2,
                                                      random_state=0)

    parameters = {
        'criterion': ('entropy', 'gini'),
        'max_depth': range(1, 21, 5),
        'ccp_alpha': np.linspace(0, 5e-2, 21),
    }
    estimator = DecisionTreeClassifier(random_state=0)
    clf = GridSearchCV(estimator=estimator,
                       param_grid=parameters,
                       refit=True,
                       n_jobs=-1,
                       cv=10)
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    val_accuracy = clf.score(X_val, y_val)

    print('Train accuracy:      {}'.format(train_accuracy))
    print('Validation accuracy: {}'.format(val_accuracy))
    print('Best parameters:')
    print(clf.best_params_)

    final_clf = \
        DecisionTreeClassifier(**clf.best_params_).fit(X_dataset, y_dataset)
    print('Final train accuracy: '
          '{}'.format(final_clf.score(X_dataset, y_dataset)))

    X_testset = ds.load_test_set()
    X_testset = preprocess_dataset(X_testset)
    test_predictions = final_clf.predict(X_testset)
    X_testset = X_testset.assign(**{ds.LABEL_COLUMN_NAME: test_predictions})
    sm.output_submission_file(X_testset, notes='trees')


if __name__ == '__main__':
    main()
