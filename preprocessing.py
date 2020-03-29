import re
from collections import Counter

import numpy as np
import pandas as pd


def convert_attribute_to_categorical(df, attribute_name):
    categories = df[attribute_name].unique()
    df[attribute_name] = pd.Categorical(df[attribute_name],
                                        categories=categories,
                                        ordered=False)
    return df


def add_title_column(df):
    name_template = r'[^,]+, (?P<title>[^\.]+)\.\s'
    name_pattern = re.compile(name_template)

    titles_list = []
    for name in df['Name']:
        match = name_pattern.match(name)
        if not match:
            titles_list.append('Unknown')
            continue
        else:
            titles_list.append(match.group('title').split(' ')[-1])

    titles_counter = Counter(titles_list)

    df['Title'] = pd.Categorical(titles_list,
                                 categories=titles_counter.keys(),
                                 ordered=False)
    return df


def add_coarse_title_column(df):
    name_template = r'[^,]+, (?P<title>[^\.]+)\.\s'
    name_pattern = re.compile(name_template)

    titles_list = []
    for name in df['Name']:
        match = name_pattern.match(name)
        if not match:
            titles_list.append('Unknown')
            continue
        else:
            titles_list.append(match.group('title').split(' ')[-1])

    title_mapping = {
        'Ms': 'Common',
        'Major': 'Rare',
        'Mr': 'Common',
        'Miss': 'Common',
        'Countess': 'Rare',
        'Dr': 'Rare',
        'Don': 'Rare',
        'Master': 'Common',
        'Jonkheer': 'Rare',
        'Mlle': 'Rare',
        'Col': 'Rare',
        'Mme': 'Rare',
        'Capt': 'Rare',
        'Rev': 'Rare',
        'Lady': 'Rare',
        'Mrs': 'Common',
        'Sir': 'Rare',
        'Dona': 'Rare',
        'Unknown': 'Unknown',
    }

    df['Title'] = pd.Categorical(
        map(lambda fine_title: title_mapping.get(fine_title, 'Rare'),
            titles_list),
        categories=set(title_mapping.values()),
        ordered=False)
    return df


def add_ticket_number_column(df):
    ticket_template = r'(?P<number>\d+)( (?P<add_info>.*))?$'
    ticket_pattern = re.compile(ticket_template)

    ticket_numbers = []
    ticket_additional_infos = set()
    for row_idx, ticket in enumerate(df['Ticket']):
        match = ticket_pattern.match(ticket[::-1])
        if not match:
            ticket_numbers.append(-1)
            continue
        ticket_numbers.append(int(match.group('number')[::-1]))

        add_info_match = match.group('add_info')
        if add_info_match:
            ticket_additional_infos.add(add_info_match[::-1])

    df = df.assign(TicketNumber=ticket_numbers)
    return df


def add_floor_column(df):
    cabin_full_template = r'^([A-Z] )?([A-Z]\d+\s?)+$'
    cabin_full_pattern = re.compile(cabin_full_template)

    cabin_template = r'(?P<floor>[A-Z])\d+\s?'
    cabin_pattern = re.compile(cabin_template)

    floors = []
    for cabin in df['Cabin']:
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

    df['Floor'] = pd.Categorical(floors,
                                 categories=filter(None, set(floors)),
                                 ordered=False)
    return df


def replace_nan_embarked_with_unknown(df):
    df.replace({'Embarked': {np.nan: 'Unknown'}}, inplace=True)
    return df


def replace_nan_age_with_value(df):
    df.replace({'Age': {np.nan: -1}}, inplace=True)
    return df


def replace_nan_fare_with_value(df):
    df.replace({'Fare': {np.nan: 0}}, inplace=True)
    return df


def convert_categorical_columns_to_numerical(df):
    categorical_columns = df.select_dtypes('category').columns
    df[categorical_columns] = \
        df[categorical_columns].apply(lambda x: x.cat.codes)
    return df


def sort_columns(df):
    df = df.reindex(columns=sorted(list(df.columns)))
    return df
