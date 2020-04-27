from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import dataset as ds
import preprocessing as pp
import submission as sm

'''
Just a copy of all the attributes.

attributes = [
    'Age',
    'BirthDate',
    'BirthPlace',
    'Cabin',
    'Destination',
    'Embarked',
    'FirstName',
    'Job',
    'LastName',
    'MaritalStatus',
    'Nationality',
    'Pclass',
    'Residence',
    'Sex',
    'Ticket',
    'Title',
    'UrlId',
    'TicketPrice',
    'TicketNumber',
    'BirthPlaceCountry',
    'BirthPlaceCity',
    'BirthPlaceRegion',
    'ResidenceCountry',
    'ResidenceCity',
    'ResidenceRegion',
    'DestinationCountry',
    'DestinationCity',
    'DestinationRegion',
    'CabinDeck',
    'AgeInDays',
    'KPassengerId',
    'SibSp',
    'Parch',
    'Split',
    'NumChild',
    'NumEmployee',
    'NumEmployer',
    'NumFriend',
    'NumKnows',
    'NumParent',
    'NumRelative',
    'NumSibling',
    'NumSpouse',
]
'''


def preprocess_dataset(df):
    keep_attributes = [
        # 'Age',
        # 'BirthDate',
        # 'BirthPlace',
        # 'Cabin',
        # 'Destination',
        'Embarked',
        # 'FirstName',
        'Job',
        # 'LastName',
        'MaritalStatus',
        'Nationality',
        'Pclass',
        # 'Residence',
        'Sex',
        # 'Ticket',
        'Title',
        # 'UrlId',
        'TicketPrice',
        'TicketNumber',
        'BirthPlaceCountry',
        # 'BirthPlaceCity',
        # 'BirthPlaceRegion',
        'ResidenceCountry',
        # 'ResidenceCity',
        # 'ResidenceRegion',
        'DestinationCountry',
        # 'DestinationCity',
        # 'DestinationRegion',
        'CabinDeck',
        'AgeInDays',
        # 'KPassengerId',
        # 'SibSp',
        # 'Parch',
        # 'Split',
        'NumChild',
        'NumEmployee',
        'NumEmployer',
        'NumFriend',
        'NumKnows',
        'NumParent',
        'NumRelative',
        'NumSibling',
        'NumSpouse',
    ]
    drop_attributes = set(df.columns) - set(keep_attributes)
    df = df.drop(columns=drop_attributes)

    # Missing tickets belong to crew member.
    df['TicketNumber'] = df['TicketNumber'].fillna(-1)

    # Use the median to replace missing ages.
    no_age_df = df.loc[df['AgeInDays'].isna()]
    for psgr_id, no_age_row in no_age_df.iterrows():
        pclass, gender, embarked = no_age_row[['Pclass', 'Sex', 'Embarked']]
        inferred_age = df.loc[(df['Pclass'] == pclass)
                              & (df['Sex'] == gender)
                              & (df['Embarked'] == embarked)
                              & (~df['AgeInDays'].isna())]['AgeInDays'].median()
        df.loc[psgr_id, 'AgeInDays'] = inferred_age

    attributes_to_categorical = [
        'Embarked',
        'Job',
        'MaritalStatus',
        'Nationality',
        'Pclass',
        'Sex',
        'Title',
        # 'TicketPrice',
        # 'TicketNumber',
        'BirthPlaceCountry',
        'ResidenceCountry',
        'DestinationCountry',
        'CabinDeck',
        # 'AgeInDays',
        # 'NumChild',
        # 'NumEmployee',
        # 'NumEmployer',
        # 'NumFriend',
        # 'NumKnows',
        # 'NumParent',
        # 'NumRelative',
        # 'NumSibling',
        # 'NumSpouse',
    ]
    df[attributes_to_categorical] = \
        df[attributes_to_categorical].fillna('Unknown')
    for attribute_name in attributes_to_categorical:
        df = pp.convert_attribute_to_categorical(df, attribute_name)

    df = pp.convert_categorical_columns_to_numerical(df)

    return df


def main():
    print('AdaBoost')

    X_dataset, y_dataset = ds.load_training_set()
    X_testset = ds.load_test_set()

    X_dataset = preprocess_dataset(X_dataset)
    X_testset = preprocess_dataset(X_testset)

    X_train, X_val, y_train, y_val = train_test_split(X_dataset,
                                                      y_dataset,
                                                      test_size=0.3,
                                                      random_state=0)

    parameters = {
        'base_estimator': [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=2),
            DecisionTreeClassifier(max_depth=3),
        ],
        'n_estimators': (100,),
        'learning_rate': (0.2,),
    }
    estimator = AdaBoostClassifier()
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

    test_predictions = clf.predict(X_testset)
    print('Testset: {}/{} survived'.format(sum(test_predictions),
                                           len(test_predictions)))

    X_testset[ds.LABEL_COLUMN_NAME] = test_predictions
    X_testset[ds.LABEL_COLUMN_NAME] = X_testset[ds.LABEL_COLUMN_NAME].astype(int)
    sm.output_submission_file(X_testset, notes='adaboost')


if __name__ == '__main__':
    main()
