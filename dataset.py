import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EVAL_FRACTION = 0.3

ID_COLUMN_NAME = 'PassengerId'
LABEL_COLUMN_NAME = 'Survived'

_project_path = os.path.join(os.environ['HOME'], 'kaggle', 'titanic')
_data_dir = os.path.join(_project_path, 'data')
_dataset_path = os.path.join(_data_dir, 'train.csv')


def _convert_attribute_to_categorical(df, attribute_name):
    categories = df[attribute_name].unique()
    df[attribute_name] = pd.Categorical(df[attribute_name], categories=categories, ordered=False)
    return df


def _add_title_column(dataset_df):
    name_template = r'[^,]+, (?P<title>[^\.]+)\.\s'
    name_pattern = re.compile(name_template)

    titles_list = []
    for name in dataset_df['Name']:
        match = name_pattern.match(name)
        if not match:
            titles_list.append('Unknown')
            continue
        else:
            titles_list.append(match.group('title').split(' ')[-1])

    titles_counter = Counter(titles_list)

    dataset_df['Title'] = pd.Categorical(titles_list,
                                         categories=titles_counter.keys(),
                                         ordered=False)
    return dataset_df


def _add_ticket_number_column(dataset_df):
    ticket_template = r'(?P<number>\d+)( (?P<add_info>.*))?$'
    ticket_pattern = re.compile(ticket_template)

    ticket_numbers = []
    ticket_additional_infos = set()
    for row_idx, ticket in enumerate(dataset_df['Ticket']):
        match = ticket_pattern.match(ticket[::-1])
        if not match:
            ticket_numbers.append(-1)
            continue
        ticket_numbers.append(int(match.group('number')[::-1]))

        add_info_match = match.group('add_info')
        if add_info_match:
            ticket_additional_infos.add(add_info_match[::-1])

    dataset_df = dataset_df.assign(TicketNumber=ticket_numbers)
    return dataset_df


def _add_floor_column(dataset_df):
    cabin_full_template = r'^([A-Z] )?([A-Z]\d+\s?)+$'
    cabin_full_pattern = re.compile(cabin_full_template)

    cabin_template = r'(?P<floor>[A-Z])\d+\s?'
    cabin_pattern = re.compile(cabin_template)

    floors = []
    for cabin in dataset_df['Cabin']:
        if cabin != cabin:
            # NaN value.
            floors.append('Unknown')
            continue

        # Check if the cabin format is atypical.
        if not cabin_full_pattern.match(cabin):
            floors.append('Unknown')
            continue

        # If you get here the cabin field is in the form:
        #  A123
        #  or: A123 B456 C78
        #  or: Q A123 B456 C78

        cabin_floors = {
            match.group('floor')
            for match in cabin_pattern.finditer(cabin)
        }

        if len(cabin_floors) != 1:
            raise ValueError('Only one floor per cabin expected, '
                             'but got {}'.format(len(cabin_floors)))
        floors.append(cabin_floors.pop())

    dataset_df['Floor'] = pd.Categorical(floors,
                                         categories=filter(None, set(floors)),
                                         ordered=False)
    return dataset_df


def _format_dataset(dataset_df):
    # Replace NaNs with valid values.
    dataset_df.replace({'Embarked': {np.nan: 'Unknown'}}, inplace=True)
    dataset_df.replace({'Age': {np.nan: -1}}, inplace=True)

    attributes_to_convert = ['Sex', 'Embarked', 'Pclass']
    for attribute_name in attributes_to_convert:
        dataset_df = _convert_attribute_to_categorical(dataset_df,
                                                       attribute_name)

    dataset_df = _add_title_column(dataset_df)
    dataset_df = _add_ticket_number_column(dataset_df)
    dataset_df = _add_floor_column(dataset_df)

    non_numerical_columns = ['Name', 'Ticket', 'Cabin']
    dataset_df.drop(columns=non_numerical_columns, inplace=True)

    return dataset_df


def get_dataset():
    dataset_df = pd.read_csv(_dataset_path)
    return dataset_df


def get_formatted_dataset():
    dataset_df = get_dataset()
    formatted_dataset_df = _format_dataset(dataset_df)
    return formatted_dataset_df


def get_numerical_dataset():
    formatted_dataset = get_formatted_dataset()
    categorical_columns = formatted_dataset.select_dtypes('category').columns
    formatted_dataset[categorical_columns] = formatted_dataset[categorical_columns].apply(lambda x: x.cat.codes)
    return formatted_dataset


def get_formatted_splits():
    dataset_df = get_formatted_dataset()
    train_df, eval_df = train_test_split(dataset_df,
                                         test_size=EVAL_FRACTION,
                                         random_state=0)
    return train_df, eval_df
